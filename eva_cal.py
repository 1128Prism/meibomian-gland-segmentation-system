import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import networkx as nx
from scipy.spatial.distance import pdist
import os


# ----------------------------
# 计算腺体平均宽度（核心改动）
# ----------------------------
def compute_gland_width(gland_mask):

    gland_mask = (gland_mask > 0).astype(np.uint8)

    dist = cv2.distanceTransform(gland_mask, cv2.DIST_L2, 5)

    skeleton = skeletonize(gland_mask).astype(np.uint8)

    width_values = dist[skeleton > 0]

    if len(width_values) == 0:
        return 0

    mean_width = 2 * np.mean(width_values)

    return mean_width

def detect_branch(skeleton, min_branch_length):
    h, w = skeleton.shape
    endpoints = []

    # 找端点（邻域=1）
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                neighbors = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2]) - 1
                if neighbors == 1:
                    endpoints.append((y, x))

    # 少于3个端点，不可能是分叉
    if len(endpoints) < 3:
        return False

    # 计算端点两两之间的最大距离（认为是主干）
    max_dist = 0
    main_pair = None

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            d = np.linalg.norm(np.array(endpoints[i]) - np.array(endpoints[j]))
            if d > max_dist:
                max_dist = d
                main_pair = (endpoints[i], endpoints[j])

    # 剩余端点是候选分支
    for pt in endpoints:
        if pt not in main_pair:
            # 到主干两端的最小距离
            d1 = np.linalg.norm(np.array(pt) - np.array(main_pair[0]))
            d2 = np.linalg.norm(np.array(pt) - np.array(main_pair[1]))

            if min(d1, d2) >= min_branch_length:
                return True

    return False

# ----------------------------
# 计算弯曲率
# ----------------------------
def compute_curvature(skeleton):

    coords = np.column_stack(np.where(skeleton > 0))

    if len(coords) < 2:
        return 1.0

    length = len(coords)

    p1 = coords[0]
    p2 = coords[-1]

    straight_dist = np.linalg.norm(p1 - p2)

    if straight_dist == 0:
        return 1.0

    return length / straight_dist


# ----------------------------
# 统计分叉点
# ----------------------------
def count_branch_points(skeleton):

    h, w = skeleton.shape
    branch_count = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
                if neighbors >= 3:
                    branch_count += 1

    return branch_count


# ----------------------------
# 主评估函数
# ----------------------------
def evaluate_meibomian_glands(original_path, gland_mask_path, conjunctiva_mask_path):
    original = cv2.imread(original_path)
    gland_mask = cv2.imread(gland_mask_path, 0)
    conjunctiva_mask = cv2.imread(conjunctiva_mask_path, 0)

    gland_mask = (gland_mask > 0).astype(np.uint8)
    conjunctiva_mask = (conjunctiva_mask > 0).astype(np.uint8)

    vis_img = original.copy()

    # ---------- 整体缺失评分 ----------
    gland_area = np.sum(gland_mask)
    conjunctiva_area = np.sum(conjunctiva_mask)

    ratio = gland_area / conjunctiva_area

    if ratio > 2/3:
        loss_level = 1
    elif ratio > 1/3:
        loss_level = 2
    else:
        loss_level = 3

    labeled = label(gland_mask)
    regions = regionprops(labeled)

    MIN_AREA = 200  # 可以调成 150~300

    filtered_regions = []
    for region in regions:
        if region.area >= MIN_AREA:
            filtered_regions.append(region)

    for region in filtered_regions:
        single_mask = (labeled == region.label).astype(np.uint8)

        skeleton = skeletonize(single_mask).astype(np.uint8)

        width = compute_gland_width(single_mask)
        curvature = compute_curvature(skeleton)
        branch_points = count_branch_points(skeleton)

        coords = region.coords

        # ----------------------------
        # 判定优先级（严格）
        # ----------------------------

        # 1️⃣ 分叉（>=3）
        is_branch = detect_branch(skeleton, min_branch_length=120)
        if is_branch:
            color = (0, 255, 255)  # 黄色

        # 2️⃣ 扭曲
        elif curvature > 1.8:
            color = (0, 0, 255)  # 红色

        # 3️⃣ 萎缩（宽度 < 2 像素）
        elif width < 2:
            color = (255, 0, 0)  # 蓝色

        # 4️⃣ 白色节段（亮度异常）
        else:
            gland_pixels = original[coords[:, 0], coords[:, 1]]
            gray_vals = cv2.cvtColor(gland_pixels.reshape(-1,1,3), cv2.COLOR_BGR2GRAY).flatten()

            if np.mean(gray_vals) > np.mean(original) + 1.5*np.std(original):
                color = (255, 0, 255)  # 紫色
            else:
                color = (0, 255, 0)  # 正常绿色

        # vis_img[coords[:, 0], coords[:, 1]] = color
        alpha = 0.4  # 透明度，0.3~0.5 比较自然

        for y, x in coords:
            original_pixel = vis_img[y, x]
            blended = (1 - alpha) * original_pixel + alpha * np.array(color)
            vis_img[y, x] = blended.astype(np.uint8)

    return vis_img, loss_level


def run_full_analysis(original_path, gland_mask_path, conjunctiva_mask_path, save_path):
    vis_img, grade = evaluate_meibomian_glands(
        original_path,
        gland_mask_path,
        conjunctiva_mask_path
    )

    analysis = ''

    cv2.imwrite(save_path, vis_img)
    return {
        "img": "/static/results/" + os.path.basename(save_path),
        "grade": grade,
        "analysis" : analysis
    }


# ==============================
# 使用示例
# ==============================
if __name__ == '__main__':
    result, grade = evaluate_meibomian_glands(
        "meibomian_original.png",
        "meibomian_mask.png",
        "meibomian_tarsal.png"
    )

    output_path = "evaluation_result.png"
    cv2.imwrite(output_path, result)

    print("Evaluation Finished")
    print(f"Dropout Grade: {grade}")
    print(f"Saved to: {output_path}")