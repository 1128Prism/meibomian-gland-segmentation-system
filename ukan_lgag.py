import torch
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd
from timm.models.helpers import named_apply
from functools import partial
from timm.models.layers import trunc_normal_tf_

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.building_blocks.archs import KANBlock

class PatchEmbed2D(nn.Module):
    """
    Encoder-safe PatchEmbed:
    - stride = 1
    - in_ch == out_ch
    - does NOT change spatial size
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,      # 关键：不能下采样
            padding=1,
            bias=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(
                g1,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer




class UKAN_UNet_2D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[nn.Conv2d],
                 kernel_sizes,
                 strides,
                 n_conv_per_stage,
                 num_classes: int,
                 n_conv_per_stage_decoder,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()

        # -------- Encoder (official nnUNet) --------
        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage,
            conv_op, kernel_sizes, strides, n_conv_per_stage,
            conv_bias, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first
        )

        # -------- KAN insertion points --------
        c_stage4 = self.encoder.output_channels[-2]
        c_bottleneck = self.encoder.output_channels[-1]

        self.lgag_4 = LGAG(
            F_g=c_bottleneck,
            F_l=c_stage4,
            F_int=c_stage4 // 2,
            kernel_size=3
        )

        self.patch_embed4 = PatchEmbed2D(c_stage4)
        self.patch_embed_bn = PatchEmbed2D(c_bottleneck)

        self.kan_stage4 = nn.ModuleList([
            KANBlock(c_stage4) for _ in range(2)
        ])
        self.kan_bn = nn.ModuleList([
            KANBlock(c_bottleneck) for _ in range(2)
        ])

        self.norm_stage4 = nn.LayerNorm(c_stage4)
        self.norm_bn = nn.LayerNorm(c_bottleneck)

        # -------- Decoder (official, unchanged) --------
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first
        )

    def forward(self, x):
        skips = self.encoder(x)

        # ---- Stage 4 KAN ----
        B, C, H, W = skips[-2].shape
        x4 = self.lgag_4(
            g=skips[-1],  # Tensor
            x=skips[-2]  # Tensor
        )
        t, h, w = self.patch_embed4(x4)
        # t, h, w = self.patch_embed4(skips[-2])
        for blk in self.kan_stage4:
            t = blk(t, h, w)
        t = self.norm_stage4(t)
        skips[-2] = t.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

        # ---- Bottleneck KAN ----
        B, C, H, W = skips[-1].shape
        t, h, w = self.patch_embed_bn(skips[-1])
        for blk in self.kan_bn:
            t = blk(t, h, w)
        t = self.norm_bn(t)
        skips[-1] = t.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

