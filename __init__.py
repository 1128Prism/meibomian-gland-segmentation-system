import csv
import json
import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, Response, jsonify, send_file
from flask_cors import *
import cv2
import copy
import torch
import shutil
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor
from eva_cal import run_full_analysis

os.environ["nnUNet_raw"] = "./nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnUNet_preprocessed"
os.environ["nnUNet_results"] = "./nnUNet_results"
import nnunetv2

# 配置Flask路由，使得前端可以访问服务器中的静态资源
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'gif', 'tif'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")

NNUNET_BASE = os.path.join(STATIC_DIR, "nnunet")
UPLOAD_DIR = os.path.join(NNUNET_BASE, "input")
GLAND_OUT = os.path.join(NNUNET_BASE, "gland_pred")
CONJUNCTIVA_OUT = os.path.join(NNUNET_BASE, "conjunctiva_pred")
RESULT_DIR = os.path.join(NNUNET_BASE, "results")

for folder in [
    NNUNET_BASE, UPLOAD_DIR, RESULT_DIR
]:
    os.makedirs(folder, exist_ok=True)

global src_img, pic_path, res_pic_path, message_get, pic_name, final


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 主页
@app.route('/')
def hello():
    return render_template('main.html')


@app.errorhandler(404)
def miss404(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def miss500(e):
    return render_template('errors/500.html'), 500


@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/model')
def model_index():
    return render_template('model/model.html')


@app.route('/transUnet2D')
def model_transUnet2D():
    return render_template('model/transUnet2D.html')


@app.route('/nnUkan')
def model_nnUkan():
    return render_template('model/nnUkan.html')


# 实时体验
@app.route('/live')
def upload_test():
    global pic_path, res_pic_path
    pic_path = 'assets/img/svg/illustration-3.svg'
    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route("/download")
def download():
    filename = request.args.get("file")
    print(filename)

    if not filename:
        return "缺少文件名", 404

    file_path = os.path.join(RESULT_DIR, filename)

    if not os.path.exists(file_path):
        return "文件不存在", 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename  # 用原始名_Seg
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("images")
    # =========================
    # 清空所有目录
    # =========================
    for folder in [UPLOAD_DIR, GLAND_OUT, CONJUNCTIVA_OUT, RESULT_DIR]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    filenames = []

    # =========================
    # 保存所有输入图像
    # =========================
    for idx, file in enumerate(files):
        original_name = os.path.basename(file.filename)

        # 去掉后缀
        base_name = os.path.splitext(original_name)[0]
        base_name = base_name.replace(" ", "_")

        # =========================
        # nnUNet标准命名
        # =========================
        name = f"{base_name}_0000.png"
        save_path = os.path.join(UPLOAD_DIR, name)
        file.save(save_path)
        filenames.append(name)

    # =========================
    # 睑板腺预测
    # =========================
    subprocess.run([
        "nnUNetv2_predict",
        "-i", UPLOAD_DIR,
        "-o", GLAND_OUT,
        "-d", "Dataset1000_MGD1k",
        "-c", "2d",
        "-f", "all"
    ], check=True)

    # =========================
    # 结膜预测
    # =========================
    subprocess.run([
        "nnUNetv2_predict",
        "-i", UPLOAD_DIR,
        "-o", CONJUNCTIVA_OUT,
        "-d", "Dataset1002_MGDCON",
        "-c", "2d",
        "-f", "all"
    ], check=True)

    # =========================
    # 腺体分析
    # =========================
    results = []

    for name in filenames:

        if not name:
            continue  # ✅ 防止 None

        base_name = name.replace("_0000.png", "")
        mask_name = base_name + ".png"
        result_filename = base_name + "_Seg.png"

        origin_path = os.path.join(UPLOAD_DIR, name)
        gland_mask = os.path.join(GLAND_OUT, mask_name)
        conjunctiva_mask = os.path.join(CONJUNCTIVA_OUT, mask_name)
        result_path = os.path.join(RESULT_DIR, result_filename)

        result = run_full_analysis(
            origin_path,
            gland_mask,
            conjunctiva_mask,
            result_path
        )

        if result is None:
            print("⚠️ 分析失败:", name)
            continue  # ✅ 不要 append None 结果

        results.append({
            "image": f"/static/nnunet/results/{result_filename}",
            "level": result.get('grade') if isinstance(result, dict) else result,
            "analysis": result.get('analysis') if isinstance(result, dict) else "",
            "filename": result_filename
        })

    return jsonify(results)


if __name__ == '__main__':
    app.run(port=5025, debug=True)
