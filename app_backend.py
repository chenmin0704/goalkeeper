"""
Goalkeeper Pose Analytics - Flask 后端 API + 前端服务
提供 YOLOv8-Pose 推理服务 + 静态前端页面

使用方法:
    1. 安装依赖: pip install ultralytics flask flask-cors opencv-python numpy
    2. 修改 MODEL_PATH 为你的模型路径
    3. 运行: python app_backend.py
    4. 浏览器打开: http://127.0.0.1:5000

API接口:
    POST /api/predict        - 上传图像进行推理
    POST /api/predict_video  - 上传视频进行推理
    GET  /api/health         - 健康检查
"""

import os
import sys
import json
import math
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ==================== 配置 ====================
MODEL_PATH = r"C:\Users\chen3\Desktop\ok\best.pt"
CONF_THRESHOLD = 0.3
# =============================================

# 构建Flask应用，同时提供静态文件和API
app = Flask(__name__, static_folder=None)
CORS(app)

# 全局模型变量
model = None
model_loaded = False

# 关键点名称（顺序固定，与YOLOv8-Pose输出一致）
KEYPOINT_NAMES = [
    'head', 'shoulder-left', 'shoulder-right',
    'hand-left', 'hand-right', 'knee-left',
    'knee-right', 'foot-left', 'foot-right'
]


def load_model():
    """加载YOLOv8-Pose模型"""
    global model, model_loaded
    if model is not None:
        return True
    try:
        from ultralytics import YOLO
        print(f"正在加载模型: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"警告: 模型文件不存在: {MODEL_PATH}")
            print("请修改 MODEL_PATH 为你的模型路径")
            return False
        model = YOLO(MODEL_PATH)
        model_loaded = True
        print("模型加载成功!")
        return True
    except ImportError:
        print("警告: ultralytics 未安装，运行: pip install ultralytics")
        return False
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def extract_keypoints(result):
    """从YOLOv8预测结果中提取关键点"""
    if result.keypoints is None:
        return None
    kpts = result.keypoints
    if kpts.xy is None or len(kpts.xy) == 0:
        return None

    xy = kpts.xy[0].cpu().numpy()
    conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else np.ones(9)

    keypoints = []
    for i in range(9):
        x, y = float(xy[i][0]), float(xy[i][1])
        c = float(conf[i]) if i < len(conf) else 1.0
        keypoints.append({
            'x': round(x, 4),
            'y': round(y, 4),
            'conf': round(c, 4),
            'name': KEYPOINT_NAMES[i]
        })
    return keypoints


def compute_bbox(keypoints, img_w, img_h):
    """基于关键点计算边界框"""
    if not keypoints:
        return [0.25, 0.1, 0.5, 0.75]
    xs = [kp['x'] for kp in keypoints]
    ys = [kp['y'] for kp in keypoints]
    margin = 20
    x_min = max(0, min(xs) - margin)
    y_min = max(0, min(ys) - margin)
    x_max = min(img_w, max(xs) + margin)
    y_max = min(img_h, max(ys) + margin)
    return [
        round(x_min / img_w, 6), round(y_min / img_h, 6),
        round((x_max - x_min) / img_w, 6), round((y_max - y_min) / img_h, 6)
    ]


def compute_body_angle(keypoints):
    """计算身体躯干角度"""
    ls = keypoints[1]
    rs = keypoints[2]
    lk = keypoints[5]
    rk = keypoints[6]

    if ls['conf'] > CONF_THRESHOLD and rs['conf'] > CONF_THRESHOLD:
        shoulder_cx = (ls['x'] + rs['x']) / 2
        shoulder_cy = (ls['y'] + rs['y']) / 2
    elif ls['conf'] > CONF_THRESHOLD:
        shoulder_cx, shoulder_cy = ls['x'], ls['y']
    elif rs['conf'] > CONF_THRESHOLD:
        shoulder_cx, shoulder_cy = rs['x'], rs['y']
    else:
        return 0.0

    if lk['conf'] > CONF_THRESHOLD and rk['conf'] > CONF_THRESHOLD:
        knee_cx = (lk['x'] + rk['x']) / 2
        knee_cy = (lk['y'] + rk['y']) / 2
    elif lk['conf'] > CONF_THRESHOLD:
        knee_cx, knee_cy = lk['x'], lk['y']
    elif rk['conf'] > CONF_THRESHOLD:
        knee_cx, knee_cy = rk['x'], rk['y']
    else:
        return 0.0

    dx = knee_cx - shoulder_cx
    dy = knee_cy - shoulder_cy
    return round(math.degrees(math.atan2(dx, abs(dy) + 1e-6)), 2)


def classify_pose(keypoints):
    """分类守门员姿态 - 降低站立准备权重50%"""
    if not keypoints:
        return "无法判断", 0.0, "未检测到关键点"

    head = keypoints[0]
    lh = keypoints[3]
    rh = keypoints[4]
    lk = keypoints[5]
    rk = keypoints[6]
    lf = keypoints[7]
    rf = keypoints[8]

    visible_count = sum(1 for kp in keypoints if kp['conf'] > CONF_THRESHOLD)
    if visible_count < 4:
        return "无法判断", 0.3, "关键点不足"

    body_angle = compute_body_angle(keypoints)

    hand_high = False
    hand_spread = False
    if lh['conf'] > CONF_THRESHOLD and rh['conf'] > CONF_THRESHOLD:
        hand_cy = (lh['y'] + rh['y']) / 2
        if head['conf'] > CONF_THRESHOLD and hand_cy < head['y'] + 20:
            hand_high = True
        if abs(lh['x'] - rh['x']) > 100:
            hand_spread = True

    # ========== 修改：降低站立准备的判定优先级 ==========

    # 只有当身体角度很小、双手未展开、且姿态很稳定时才判定为站立
    is_standing = (
        abs(body_angle) <= 10 and  # 严格限制：角度必须很小
        not hand_spread and         # 双手未展开
        not hand_high and           # 双手未高举
        visible_count >= 7          # 关键点足够多（姿态稳定）
    )

    if is_standing:
        pose = "站立准备"
        confidence = 0.3  # 降低置信度（原来是0.5）
        details = [f"身体直立 ({body_angle:.1f}度)"]
    else:
        # 优先判定为扑救动作
        if abs(body_angle) > 45:
            pose = "向右倒地扑救" if body_angle > 0 else "向左倒地扑救"
            confidence = min(abs(body_angle) / 90, 1.0)
            details = [f"{'向右' if body_angle > 0 else '向左'}倒地 ({body_angle:.1f}度)"]
        elif abs(body_angle) > 20:
            pose = "向右侧扑" if body_angle > 0 else "向左侧扑"
            confidence = abs(body_angle) / 60
            details = [f"{'向右' if body_angle > 0 else '向左'}侧扑 ({body_angle:.1f}度)"]
        else:
            # 角度不大但也不满足站立条件，判定为准备动作
            pose = "扑救准备"
            confidence = 0.4
            details = [f"身体微倾 ({body_angle:.1f}度)"]

    if hand_high and hand_spread and pose in ["站立准备", "扑救准备"]:
        pose = "高空接球"
        details.append("双手高举")
    elif hand_spread:
        details.append("单手伸展扑救" if "倒地" in pose or "侧扑" in pose else "双手展开准备")

    # 再次降低站立准备的置信度（权重减少50%）
    if pose == "站立准备":
        confidence = confidence * 0.5  # 权重减半

    return pose, round(confidence, 4), "; ".join(details) if details else "常规姿态"


def predict_direction(keypoints):
    """预测扑救方向"""
    lh = keypoints[3]
    rh = keypoints[4]
    head = keypoints[0]
    if lh['conf'] < CONF_THRESHOLD or rh['conf'] < CONF_THRESHOLD:
        return "未知", 0.0
    hand_cx = (lh['x'] + rh['x']) / 2
    if head['conf'] > CONF_THRESHOLD:
        diff = hand_cx - head['x']
        if diff < -40:
            return "左侧", min(abs(diff) / 100, 1.0)
        elif diff > 40:
            return "右侧", min(abs(diff) / 100, 1.0)
    return "正中", 0.5


def analyze_coverage(keypoints, img_w, img_h):
    """分析覆盖范围"""
    visible_pts = [(kp['x'], kp['y']) for kp in keypoints if kp['conf'] > CONF_THRESHOLD]
    if len(visible_pts) < 3:
        return 0.0, "关键点不足"
    xs = [p[0] for p in visible_pts]
    ys = [p[1] for p in visible_pts]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    img_area = img_w * img_h
    coverage = min(area / img_area * 100, 100)
    desc = "大幅伸展" if coverage > 15 else "中等范围" if coverage > 8 else "紧凑姿态"
    return round(coverage, 2), desc


# ==================== API 路由 ====================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded, 'model_path': MODEL_PATH})


@app.route('/api/predict', methods=['POST'])
def predict_image():
    if not model_loaded:
        return jsonify({
            'error': '模型未加载',
            'detail': '请检查模型路径是否正确，或重启后端服务',
            'model_path': MODEL_PATH
        }), 500
    if 'image' not in request.files:
        return jsonify({'error': '请上传图像文件（字段名: image）'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    tmp_path = None
    try:
        ext = os.path.splitext(secure_filename(file.filename))[1] or '.jpg'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        if img is None:
            return jsonify({
                'error': '无法读取图像',
                'detail': '图像格式不支持或文件已损坏',
                'filename': file.filename
            }), 400

        img_h, img_w = img.shape[:2]
        results = model(tmp_path, conf=CONF_THRESHOLD)
        result = results[0]
        keypoints = extract_keypoints(result)

        if keypoints is None:
            return jsonify({
                'error': '未检测到守门员',
                'detail': '图像中没有检测到守门员，可能原因：1) 画面中无守门员 2) 画面模糊 3) 角度不合适',
                'suggestion': '尝试更换更清晰的守门员图片'
            }), 404

        bbox = compute_bbox(keypoints, img_w, img_h)
        for kp in keypoints:
            kp['x'] = round(kp['x'] / img_w, 6)
            kp['y'] = round(kp['y'] / img_h, 6)

        pose, pose_conf, detail = classify_pose(keypoints)
        direction, dir_conf = predict_direction(keypoints)
        coverage, cov_desc = analyze_coverage(keypoints, img_w, img_h)
        body_angle = compute_body_angle(keypoints)

        return jsonify({
            'success': True,
            'pose': pose,
            'pose_conf': round(pose_conf, 4),
            'direction': direction,
            'direction_conf': round(dir_conf, 4),
            'coverage': coverage,
            'coverage_desc': cov_desc,
            'detail': detail,
            'body_angle': body_angle,
            'keypoints': keypoints,
            'bbox': bbox,
            'image_size': {'width': img_w, 'height': img_h}
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': '图像分析失败',
            'detail': str(e),
            'traceback': traceback.format_exc()
        }), 500

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.route('/api/predict_video', methods=['POST'])
def predict_video():
    if not model_loaded:
        return jsonify({'error': '模型未加载', 'detail': '请检查模型路径是否正确，或重启后端服务'}), 500
    if 'video' not in request.files:
        return jsonify({'error': '请上传视频文件（字段名: video）'}), 400

    file = request.files['video']
    tmp_path = None
    cap = None

    try:
        # 检查文件是否为空
        if file.filename == '':
            return jsonify({'error': '文件名为空', 'detail': '请选择有效的视频文件'}), 400

        # 获取文件扩展名
        ext = os.path.splitext(secure_filename(file.filename))[1] or '.mp4'

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            print(f"临时视频文件: {tmp_path}, 大小: {os.path.getsize(tmp_path) / 1024 / 1024:.2f} MB")

        # 打开视频
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({
                'error': '无法打开视频文件',
                'detail': '视频格式可能不支持，请尝试转换为 MP4 格式 (H.264 编码)',
                'filename': file.filename
            }), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧, 约{total_frames / fps:.1f}秒")

        # 检查视频参数是否有效
        if fps <= 0 or total_frames <= 0:
            return jsonify({
                'error': '视频参数异常',
                'detail': f'fps={fps}, total_frames={total_frames}, 视频可能已损坏',
                'filename': file.filename
            }), 400

        frames_results = []
        frame_idx = 0
        sample_interval = 1  # 逐帧分析

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 采样间隔
            if frame_idx % sample_interval != 0:
                continue

            img_h, img_w = frame.shape[:2]

            # 模型推理
            try:
                results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            except Exception as model_error:
                print(f"第{frame_idx}帧模型推理失败: {str(model_error)}")
                continue  # 跳过这一帧，继续下一帧

            result = results[0]
            keypoints = extract_keypoints(result)

            if keypoints:
                for kp in keypoints:
                    kp['x'] = round(kp['x'] / img_w, 6)
                    kp['y'] = round(kp['y'] / img_h, 6)
                pose, pose_conf, detail = classify_pose(keypoints)
                direction, dir_conf = predict_direction(keypoints)
                frames_results.append({
                    'frame': frame_idx,
                    'time': round(frame_idx / fps, 2),
                    'pose': pose,
                    'pose_conf': round(pose_conf, 4),
                    'direction': direction,
                    'direction_conf': round(dir_conf, 4),
                    'body_angle': compute_body_angle(keypoints),
                    'keypoints': keypoints
                })

        # 检查是否有分析结果
        if not frames_results:
            return jsonify({
                'error': '未检测到守门员',
                'detail': '视频中没有检测到守门员姿态，可能原因：1) 视频中无守门员 2) 画面模糊 3) 置信度阈值太高',
                'suggestion': '尝试降低 CONF_THRESHOLD 或更换更清晰的视频',
                'video_info': {
                    'filename': file.filename,
                    'width': width,
                    'height': height,
                    'fps': round(fps, 2),
                    'total_frames': total_frames,
                    'duration': round(total_frames / fps, 2) if fps > 0 else 0,
                    'analyzed_frames': frame_idx
                }
            }), 404

        # 统计姿态分布
        from collections import Counter
        pose_counts = Counter(f['pose'] for f in frames_results)

        return jsonify({
            'success': True,
            'total_frames': total_frames,
            'analyzed_frames': len(frames_results),
            'fps': round(fps, 2),
            'pose_distribution': dict(pose_counts),
            'frames': frames_results
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"视频分析异常: {str(e)}")
        print(error_trace)

        return jsonify({
            'error': '视频分析失败',
            'detail': str(e),
            'traceback': error_trace,
            'suggestion': '请检查视频格式是否为 MP4 (H.264编码)，或尝试更短视频'
        }), 500

    finally:
        # 确保资源释放
        if cap is not None:
            cap.release()
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"临时文件已清理: {tmp_path}")
            except Exception as cleanup_error:
                print(f"清理临时文件失败: {cleanup_error}")

# ==================== 前端静态文件服务 ====================

# 获取dist目录路径（与后端脚本同目录下的dist文件夹）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(SCRIPT_DIR, 'dist')

# 如果dist不存在，尝试其他可能的位置
if not os.path.exists(DIST_DIR):
    DIST_DIR = os.path.join(SCRIPT_DIR, 'app', 'dist')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    """提供前端静态文件"""
    if not path:
        path = 'index.html'

    # API路由不走静态文件
    if path.startswith('api/'):
        return jsonify({'error': 'Not Found'}), 404

    file_path = os.path.join(DIST_DIR, path)

    # 如果文件存在，直接返回
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(DIST_DIR, path)

    # 如果是目录或不存在的文件，返回index.html（支持前端路由）
    index_path = os.path.join(DIST_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(DIST_DIR, 'index.html')

    # dist目录不存在时的提示
    return """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Goalkeeper Analytics</title></head>
<body style="font-family:sans-serif;max-width:600px;margin:50px auto;padding:20px">
<h2>Goalkeeper Pose Analytics - 后端运行中</h2>
<p style="color:green">Flask API 服务正常!</p>
<h3>后端API可用:</h3>
<ul>
<li>GET <a href="/api/health">/api/health</a> - 健康检查</li>
<li>POST /api/predict - 图像推理</li>
<li>POST /api/predict_video - 视频推理</li>
</ul>
<h3>前端部署方式（选其一）:</h3>
<ol>
<li><b>在线访问（推荐）:</b> <a href="https://4h5pe6ethhyj4.ok.kimi.link/">https://4h5pe6ethhyj4.ok.kimi.link/</a></li>
<li><b>本地前端 + 本地后端:</b><br>
   将前端dist文件夹复制到与本脚本同级目录，然后刷新此页面</li>
</ol>
<p style="color:#666;font-size:12px">模型路径: {model_path}<br>模型状态: {model_status}</p>
</body>
</html>""".format(
        model_path=MODEL_PATH,
        model_status="已加载" if model_loaded else "未加载"
    ), 200


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("  Goalkeeper Pose Analytics - Flask 后端")
    print("=" * 60)
    print()

    load_model()
    print()
    print("启动 Flask 服务...")
    print("访问地址: http://127.0.0.1:5000")
    print()
    print("API 接口:")
    print("  GET  /api/health         - 健康检查")
    print("  POST /api/predict        - 图像推理")
    print("  POST /api/predict_video  - 视频推理")
    print()

    if os.path.exists(DIST_DIR):
        print(f"前端静态文件目录: {DIST_DIR}")
    else:
        print(f"前端dist目录未找到: {DIST_DIR}")
        print("前端将显示提示页面，或使用在线地址")
    print()

    app.run(host='0.0.0.0', port=5000, debug=False)