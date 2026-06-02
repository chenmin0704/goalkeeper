from flask import Flask, render_template_string, request, jsonify
import os
import sys
import math
import tempfile
from collections import Counter
from pathlib import Path

# ===================== 模型路径自动搜索 =====================
# 根据你的训练日志，模型保存在 goalkeeper_pose2 文件夹
# 代码会自动搜索以下路径，找到可用的模型文件

POSSIBLE_MODEL_PATHS = [
    # 基于你训练日志的实际路径
    r"C:\Users\chen3\Desktop\ok\新建文件夹\runs\goalkeeper_pose2\weights\best.pt",
    # 也尝试旧的命名（兼容性）
    r"C:\Users\chen3\Desktop\ok\新建文件夹\runs\goalkeeper_pose\weights\best.pt",
    # 相对路径（如果脚本和runs在同一目录）
    r".\runs\goalkeeper_pose2\weights\best.pt",
    r".\runs\goalkeeper_pose\weights\best.pt",
    r"runs\goalkeeper_pose2\weights\best.pt",
    r"runs\goalkeeper_pose\weights\best.pt",
    # 上级目录
    r"..\runs\goalkeeper_pose2\weights\best.pt",
    r"..\runs\goalkeeper_pose\weights\best.pt",
]

MODEL_PATH = None
for p in POSSIBLE_MODEL_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

# 如果自动搜索失败，尝试遍历 runs 目录下的所有 best.pt
if MODEL_PATH is None:
    for base in [r"C:\Users\chen3\Desktop\ok\新建文件夹", r"."]:
        runs_dir = os.path.join(base, "runs")
        if os.path.isdir(runs_dir):
            for root, dirs, files in os.walk(runs_dir):
                if "best.pt" in files:
                    MODEL_PATH = os.path.join(root, "best.pt")
                    break
            if MODEL_PATH:
                break

# ===================== 初始化 Flask =====================
app = Flask(__name__)

# ===================== 尝试加载模型 =====================
model = None
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"[INFO] 正在加载模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print(f"[OK] 模型加载成功！")
    else:
        print("[ERROR] 未找到模型文件 best.pt")
        print("[HINT] 请确认训练生成的 runs/goalkeeper_pose2/weights/best.pt 存在")
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    print("[HINT] 请安装依赖: pip install ultralytics opencv-python flask")
except Exception as e:
    print(f"[ERROR] 模型加载失败: {e}")

KEYPOINT_NAMES = ['头', '左肩', '右肩', '左手', '右手', '左膝', '右膝', '左脚', '右脚']
COLORS = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', '#45B7D1', '#96CEB4', '#96CEB4', '#FFEAA7', '#FFEAA7']
SKELETON = [[0, 1], [0, 2], [1, 3], [2, 4], [1, 5], [2, 6], [5, 7], [6, 8], [1, 2], [5, 6]]


def analyze(keypoints):
    """分析守门员姿态"""
    ls, rs, lk, rk = keypoints[1], keypoints[2], keypoints[5], keypoints[6]

    def midpoint(a, b):
        if a[2] > 0.3 and b[2] > 0.3:
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return (a[0], a[1]) if a[2] > 0.3 else (b[0], b[1]) if b[2] > 0.3 else None

    shoulder = midpoint(ls, rs)
    knee = midpoint(lk, rk)

    body_angle = 0
    if shoulder and knee:
        body_angle = math.degrees(math.atan2(knee[0] - shoulder[0], abs(knee[1] - shoulder[1]) + 1e-6))

    if abs(body_angle) > 45:
        pose = "向右倒地扑救" if body_angle > 0 else "向左倒地扑救"
    elif abs(body_angle) > 20:
        pose = "向右侧扑" if body_angle > 0 else "向左侧扑"
    elif ls[2] > 0.3 and rs[2] > 0.3 and keypoints[3][2] > 0.3 and keypoints[4][2] > 0.3:
        hand_cy = (keypoints[3][1] + keypoints[4][1]) / 2
        if hand_cy < keypoints[0][1] + 20 and abs(keypoints[3][0] - keypoints[4][0]) > 100:
            pose = "高空接球"
        else:
            pose = "站立准备"
    else:
        pose = "站立准备"

    if keypoints[3][2] > 0.3 and keypoints[4][2] > 0.3 and keypoints[0][2] > 0.3:
        hand_cx = (keypoints[3][0] + keypoints[4][0]) / 2
        diff = hand_cx - keypoints[0][0]
        direction = "左侧" if diff < -40 else "右侧" if diff > 40 else "正中"
    else:
        direction = "正中"

    return pose, body_angle, direction


def extract_keypoints(result, w, h):
    """从YOLO结果提取关键点并归一化"""
    if result.keypoints is None:
        return None
    kpts = result.keypoints
    if len(kpts.xy) == 0:
        return None
    xy = kpts.xy[0].cpu().numpy()
    conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else np.ones(len(xy))
    if len(xy) < 9:
        return None
    kps = [[float(xy[i][0]) / w, float(xy[i][1]) / h, float(conf[i])] for i in range(9)]
    return kps


@app.route('/')
def index():
    if model is None:
        return render_template_string(ERROR_HTML)
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未加载，请检查服务器日志'}), 503

    if 'image' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '空文件'}), 400

    try:
        suffix = os.path.splitext(file.filename)[1] or '.jpg'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        os.unlink(tmp_path)

        if img is None:
            return jsonify({'error': '无法读取图像文件'}), 400

        h, w = img.shape[:2]
        results = model(img, conf=0.3)

        if not results or len(results) == 0:
            return jsonify({'error': '模型未返回结果'})

        kps = extract_keypoints(results[0], w, h)
        if kps is None:
            return jsonify({'error': '未检测到守门员，请确保图像中包含守门员'})

        pose, angle, direction = analyze(kps)
        return jsonify({
            'keypoints': kps,
            'pose': pose,
            'angle': round(angle, 1),
            'direction': direction,
            'names': KEYPOINT_NAMES,
            'skeleton': SKELETON,
            'colors': COLORS
        })
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/predict_video', methods=['POST'])
def predict_video():
    if model is None:
        return jsonify({'error': '模型未加载'}), 503

    if 'video' not in request.files:
        return jsonify({'error': '没有上传视频文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '空文件'}), 400

    try:
        ext = os.path.splitext(file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            os.unlink(tmp_path)
            return jsonify({'error': '无法打开视频文件'}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0
        interval = max(1, int(fps / 5))  # 每秒约5帧

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % interval != 0:
                continue

            h, w = frame.shape[:2]
            results = model(frame, conf=0.3, verbose=False)

            if results and len(results) > 0:
                kps = extract_keypoints(results[0], w, h)
                if kps:
                    pose, angle, direction = analyze(kps)
                    frames.append({
                        'frame': frame_idx,
                        'time': round(frame_idx / fps, 2),
                        'pose': pose,
                        'angle': round(angle, 1),
                        'direction': direction
                    })

        cap.release()
        os.unlink(tmp_path)

        # 统计
        pose_counts = Counter(f['pose'] for f in frames)
        direction_counts = Counter(f['direction'] for f in frames)

        # 检测扑救时刻
        save_moments = []
        in_action = False
        start_frame = 0
        for f in frames:
            is_save = f['pose'] not in ['站立准备', '高空接球']
            if is_save and not in_action:
                in_action = True
                start_frame = f['frame']
            elif not is_save and in_action:
                in_action = False
                if f['frame'] - start_frame >= interval * 2:
                    save_moments.append({
                        'start_frame': start_frame,
                        'end_frame': f['frame'],
                        'time': round(start_frame / fps, 2)
                    })

        return jsonify({
            'total_frames': total,
            'fps': round(fps, 1),
            'analyzed_frames': len(frames),
            'pose_distribution': dict(pose_counts),
            'direction_distribution': dict(direction_counts),
            'save_moments': save_moments,
            'frames': frames
        })
    except Exception as e:
        return jsonify({'error': f'视频处理失败: {str(e)}'}), 500


# ===================== HTML 模板 =====================

ERROR_HTML = '''
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><title>模型加载失败</title>
<style>
body { font-family: sans-serif; background: #0f172a; color: #e2e8f0; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
.box { background: #1e293b; padding: 40px; border-radius: 16px; max-width: 600px; text-align: center; }
h1 { color: #f87171; font-size: 22px; }
.code { background: #0f172a; padding: 16px; border-radius: 8px; text-align: left; font-family: monospace; font-size: 13px; color: #94a3b8; margin: 16px 0; }
.btn { background: #059669; color: #fff; border: none; padding: 10px 24px; border-radius: 8px; cursor: pointer; font-size: 14px; }
</style></head>
<body>
<div class="box">
  <h1>&#9940; 模型未加载</h1>
  <p>Flask 服务器运行正常，但 YOLO 模型未能成功加载。</p>
  <div class="code">
    <strong>可能原因：</strong><br>
    1. 模型文件 best.pt 路径不正确<br>
    2. 依赖未安装: pip install ultralytics opencv-python<br>
    3. 训练生成的 runs 目录不在预期位置<br><br>
    <strong>搜索过的路径：</strong><br>
    ''' + '<br>'.join(POSSIBLE_MODEL_PATHS) + '''
  </div>
  <p>请确认训练生成的 <code>runs/goalkeeper_pose2/weights/best.pt</code> 文件存在，然后重启服务器。</p>
  <button class="btn" onclick="location.reload()">刷新重试</button>
</div>
</body></html>
'''

HTML = '''
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>守门员扑救姿态分析</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0f172a; color:#e2e8f0; min-height:100vh; }
.container { max-width:1200px; margin:0 auto; padding:20px; }
header { display:flex; align-items:center; gap:12px; margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid #334155; }
header .icon { width:40px; height:40px; background:#059669; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0; }
header h1 { font-size:20px; color:#fff; }
header p { font-size:12px; color:#64748b; margin-top:2px; }

.mode-switch { display:flex; gap:4px; background:#1e293b; padding:4px; border-radius:8px; margin-bottom:16px; width:fit-content; }
.mode-switch button { padding:8px 20px; border:none; border-radius:6px; font-size:13px; cursor:pointer; background:transparent; color:#94a3b8; transition:.2s; }
.mode-switch button.active { background:#059669; color:#fff; }

.main { display:grid; grid-template-columns: 1fr 360px; gap:20px; }
@media (max-width:960px) { .main { grid-template-columns:1fr; } }

.upload-box { border:2px dashed #475569; border-radius:12px; padding:50px 30px; text-align:center; cursor:pointer; transition:.2s; background:#1e293b; }
.upload-box:hover { border-color:#059669; }
.upload-box.dragover { border-color:#059669; background:#064e3b30; }
.upload-box input { display:none; }
.upload-box .icon-big { font-size:40px; margin-bottom:10px; opacity:.6; }
.upload-box p { font-size:14px; color:#cbd5e1; }
.upload-box .sub { font-size:12px; color:#64748b; margin-top:4px; }

.image-area { position:relative; background:#1e293b; border-radius:12px; overflow:hidden; min-height:280px; display:flex; align-items:center; justify-content:center; }
.image-area img, .image-area canvas { max-width:100%; max-height:550px; display:block; }
.image-area .placeholder { color:#475569; font-size:14px; }

#loading { position:absolute; inset:0; background:rgba(15,23,42,.85); display:none; flex-direction:column; align-items:center; justify-content:center; gap:12px; z-index:10; backdrop-filter:blur(4px); }
#loading.show { display:flex; }
.spinner { width:36px; height:36px; border:3px solid #334155; border-top-color:#34d399; border-radius:50%; animation:spin 1s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
#loading .text { font-size:13px; color:#94a3b8; }

.change-btn { position:absolute; top:10px; right:10px; background:rgba(0,0,0,.65); color:#fff; border:none; padding:6px 14px; border-radius:6px; font-size:12px; cursor:pointer; z-index:5; }

.panel { background:#1e293b; border-radius:12px; padding:16px; margin-bottom:16px; }
.panel h3 { font-size:13px; color:#64748b; margin-bottom:12px; display:flex; align-items:center; gap:6px; text-transform:uppercase; letter-spacing:.5px; }

.result-item { display:flex; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid #334155; }
.result-item:last-child { border-bottom:none; }
.result-item .label { color:#94a3b8; font-size:13px; }
.result-item .value { font-size:15px; font-weight:600; }
.value.standing { color:#34d399; }
.value.dive { color:#fb923c; }
.value.ground { color:#f87171; }
.value.catch { color:#a78bfa; }

.kp-list { margin-top:4px; }
.kp-row { display:flex; align-items:center; gap:8px; padding:5px 0; font-size:12px; }
.kp-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.kp-name { width:50px; color:#cbd5e1; }
.kp-bar { flex:1; height:5px; background:#334155; border-radius:3px; overflow:hidden; }
.kp-bar-fill { height:100%; border-radius:3px; }
.kp-val { width:36px; text-align:right; color:#64748b; font-size:11px; }

.chart-bar { display:flex; align-items:center; gap:10px; padding:6px 0; font-size:13px; }
.chart-label { width:100px; color:#94a3b8; text-align:right; flex-shrink:0; }
.chart-track { flex:1; height:22px; background:#334155; border-radius:6px; overflow:hidden; position:relative; }
.chart-fill { height:100%; border-radius:6px; display:flex; align-items:center; padding-left:8px; color:#fff; font-size:11px; font-weight:600; transition:width .5s; }
.chart-val { width:40px; color:#64748b; font-size:12px; }

.timeline { margin-top:12px; max-height:240px; overflow-y:auto; }
.timeline-row { display:flex; align-items:center; gap:10px; padding:5px 10px; border-radius:6px; font-size:12px; }
.timeline-row:nth-child(odd) { background:#0f172a40; }
.timeline-row .t-time { width:50px; color:#64748b; flex-shrink:0; }
.timeline-row .t-pose { flex:1; color:#e2e8f0; }
.timeline-row .t-dir { width:40px; text-align:center; }
.timeline-row .t-angle { width:50px; text-align:right; color:#64748b; }

.save-moments { margin-top:12px; }
.save-tag { display:inline-block; background:#dc262630; color:#fca5a5; padding:4px 10px; border-radius:6px; font-size:12px; margin:3px; }
.save-tag.green { background:#05966930; color:#6ee7b7; }

.dist-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; margin-top:8px; }
.dist-item { background:#0f172a; padding:10px; border-radius:8px; text-align:center; }
.dist-item .num { font-size:20px; font-weight:700; color:#34d399; }
.dist-item .lbl { font-size:11px; color:#64748b; margin-top:2px; }

.empty-state { text-align:center; padding:40px; color:#475569; font-size:14px; }
</style>
</head>
<body>
<div class="container">
<header>
  <div class="icon">&#9917;</div>
  <div>
    <h1>守门员扑救姿态分析</h1>
    <p>YOLOv8-Pose 关键点检测 | 图像 / 视频分析</p>
  </div>
</header>

<div class="mode-switch">
  <button class="active" onclick="switchMode('image', this)">&#128247; 图像分析</button>
  <button onclick="switchMode('video', this)">&#127909; 视频分析</button>
</div>

<div class="main">
  <div>
    <div id="imageUpload">
      <div class="upload-box" id="uploadBox">
        <input type="file" id="fileInput" accept="image/*">
        <div class="icon-big">&#128444;</div>
        <p>点击或拖拽上传图像</p>
        <p class="sub">JPG、PNG 格式</p>
      </div>
    </div>
    <div id="videoUpload" style="display:none;">
      <div class="upload-box" id="videoUploadBox">
        <input type="file" id="videoInput" accept="video/*">
        <div class="icon-big">&#127916;</div>
        <p>点击或拖拽上传视频</p>
        <p class="sub">MP4 格式，自动逐帧分析姿态</p>
      </div>
    </div>

    <div class="image-area" id="imageArea" style="display:none;">
      <div class="placeholder" id="placeholder">分析中...</div>
      <img id="previewImg" style="display:none;">
      <canvas id="resultCanvas" style="display:none;"></canvas>
      <button class="change-btn" id="changeBtn" style="display:none;">更换</button>
      <div id="loading"><div class="spinner"></div><div class="text">分析中...</div></div>
    </div>

    <div class="panel" id="videoResult" style="display:none;">
      <h3>&#128202; 视频分析结果</h3>
      <div id="videoStats"></div>
      <div class="dist-grid" id="videoDist" style="margin-top:12px;"></div>
      <div id="saveMoments" class="save-moments"></div>
      <div class="timeline" id="videoTimeline"></div>
    </div>
  </div>

  <div>
    <div class="panel" id="imagePanel">
      <h3>&#9971; 姿态识别</h3>
      <div id="poseResult">
        <div class="empty-state">上传图像开始分析</div>
      </div>
      <div class="kp-list" id="kpList" style="display:none;"></div>
    </div>
  </div>
</div>
</div>

<script>
let currentMode = 'image';

function switchMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-switch button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('imageUpload').style.display = mode === 'image' ? 'block' : 'none';
  document.getElementById('videoUpload').style.display = mode === 'video' ? 'block' : 'none';
  document.getElementById('imageArea').style.display = mode === 'image' ? 'flex' : 'none';
  document.getElementById('imagePanel').style.display = mode === 'image' ? 'block' : 'none';
  document.getElementById('videoResult').style.display = mode === 'video' ? 'block' : 'none';
  if (mode === 'video') {
    document.getElementById('imageArea').style.display = 'none';
  }
}

const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');

uploadBox.addEventListener('click', () => fileInput.click());
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', e => { e.preventDefault(); uploadBox.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleImage(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', e => { if(e.target.files[0]) handleImage(e.target.files[0]); });
document.getElementById('changeBtn').addEventListener('click', () => fileInput.click());

function handleImage(file) {
  if (!file.type.startsWith('image/')) return alert('请上传图像文件');
  const url = URL.createObjectURL(file);
  const img = document.getElementById('previewImg');
  img.src = url;
  img.style.display = 'block';
  document.getElementById('resultCanvas').style.display = 'none';
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('uploadBox').style.display = 'none';
  document.getElementById('changeBtn').style.display = 'block';
  document.getElementById('imageArea').style.display = 'flex';

  const formData = new FormData();
  formData.append('image', file);
  document.getElementById('loading').classList.add('show');

  fetch('/predict', { method:'POST', body:formData })
    .then(async r => {
      const text = await r.text();
      try { return JSON.parse(text); }
      catch { throw new Error('服务器返回非JSON: ' + text.substring(0,200)); }
    })
    .then(data => {
      document.getElementById('loading').classList.remove('show');
      if (data.error) { alert('分析结果: ' + data.error); return; }
      drawImageResult(data);
      showImageResults(data);
    })
    .catch(e => {
      document.getElementById('loading').classList.remove('show');
      alert('请求失败: ' + e.message);
    });
}

function drawImageResult(data) {
  const img = document.getElementById('previewImg');
  const canvas = document.getElementById('resultCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  const W = canvas.width, H = canvas.height;
  const kps = data.keypoints;

  ctx.lineWidth = 3;
  data.skeleton.forEach(([a,b]) => {
    if (kps[a][2]<0.3 || kps[b][2]<0.3) return;
    ctx.strokeStyle = '#34d39990';
    ctx.beginPath();
    ctx.moveTo(kps[a][0]*W, kps[a][1]*H);
    ctx.lineTo(kps[b][0]*W, kps[b][1]*H);
    ctx.stroke();
  });

  kps.forEach((kp,i) => {
    if (kp[2]<0.3) return;
    const x=kp[0]*W, y=kp[1]*H;
    ctx.beginPath(); ctx.arc(x,y,8,0,6.28); ctx.fillStyle=data.colors[i]+'55'; ctx.fill();
    ctx.beginPath(); ctx.arc(x,y,4,0,6.28); ctx.fillStyle=data.colors[i]; ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle='#fff'; ctx.font='bold 13px sans-serif'; ctx.shadowColor='rgba(0,0,0,.8)'; ctx.shadowBlur=4;
    ctx.fillText(data.names[i], x+10, y-6); ctx.shadowBlur=0;
  });

  img.style.display = 'none';
  canvas.style.display = 'block';
}

function showImageResults(data) {
  const cls = data.pose.includes('倒地') ? 'ground' : data.pose.includes('侧扑') ? 'dive' : data.pose === '高空接球' ? 'catch' : 'standing';
  document.getElementById('poseResult').innerHTML = `
    <div class="result-item"><span class="label">姿态</span><span class="value ${cls}">${data.pose}</span></div>
    <div class="result-item"><span class="label">方向</span><span class="value" style="color:#38bdf8">${data.direction}</span></div>
    <div class="result-item"><span class="label">身体角度</span><span class="value" style="color:#fbbf24">${data.angle}\u00B0</span></div>`;

  document.getElementById('kpList').style.display = 'block';
  document.getElementById('kpList').innerHTML = data.keypoints.map((kp,i) => `
    <div class="kp-row">
      <div class="kp-dot" style="background:${data.colors[i]}"></div>
      <span class="kp-name">${data.names[i]}</span>
      <div class="kp-bar"><div class="kp-bar-fill" style="width:${Math.min(kp[2]*100,100)}%;background:${data.colors[i]}"></div></div>
      <span class="kp-val">${(kp[2]*100).toFixed(0)}%</span>
    </div>`).join('');
}

const vBox = document.getElementById('videoUploadBox');
const vInput = document.getElementById('videoInput');

vBox.addEventListener('click', () => vInput.click());
vBox.addEventListener('dragover', e => { e.preventDefault(); vBox.classList.add('dragover'); });
vBox.addEventListener('dragleave', () => vBox.classList.remove('dragover'));
vBox.addEventListener('drop', e => { e.preventDefault(); vBox.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleVideo(e.dataTransfer.files[0]); });
vInput.addEventListener('change', e => { if(e.target.files[0]) handleVideo(e.target.files[0]); });

function handleVideo(file) {
  if (!file.type.startsWith('video/')) return alert('请上传视频文件');
  vBox.innerHTML = '<div class="icon-big">&#9203;</div><p>分析中，请稍候...</p><p class="sub">正在逐帧检测关键点</p>';
  vBox.style.cursor = 'default';

  const formData = new FormData();
  formData.append('video', file);

  fetch('/predict_video', { method:'POST', body:formData })
    .then(async r => {
      const text = await r.text();
      try { return JSON.parse(text); }
      catch { throw new Error('服务器返回非JSON: ' + text.substring(0,200)); }
    })
    .then(data => {
      if (data.error) { vBox.innerHTML = '<div class="icon-big">&#10060;</div><p>分析失败</p><p class="sub">' + data.error + '</p>'; return; }
      vBox.style.display = 'none';
      showVideoResults(data);
    })
    .catch(e => {
      vBox.innerHTML = '<div class="icon-big">&#10060;</div><p>请求失败</p><p class="sub">' + e.message + '</p>';
    });
}

function showVideoResults(data) {
  const resultDiv = document.getElementById('videoResult');
  const statsDiv = document.getElementById('videoStats');
  resultDiv.style.display = 'block';

  const total = data.total_frames;
  const analyzed = data.analyzed_frames;
  const saves = data.save_moments.length;

  statsDiv.innerHTML = `
    <div class="dist-grid">
      <div class="dist-item"><div class="num">${total}</div><div class="lbl">总帧数</div></div>
      <div class="dist-item"><div class="num" style="color:#38bdf8">${analyzed}</div><div class="lbl">分析帧</div></div>
      <div class="dist-item"><div class="num" style="color:#f87171">${saves}</div><div class="lbl">扑救次数</div></div>
    </div>
  `;

  const poses = data.pose_distribution;
  const maxCount = Math.max(...Object.values(poses), 1);
  const poseColors = {'站立准备':'#34d399','高空接球':'#a78bfa','向左侧扑':'#fb923c','向右侧扑':'#fb923c',
    '向左倒地扑救':'#f87171','向右倒地扑救':'#f87171'};

  let distHtml = '<div style="margin-top:16px"><h3 style="font-size:13px;color:#64748b;margin-bottom:10px">&#128202; 姿态分布</h3>';
  for (const [pose, count] of Object.entries(poses).sort((a,b) => b[1] - a[1])) {
    const pct = ((count / analyzed) * 100).toFixed(1);
    const width = (count / maxCount * 100).toFixed(1);
    const color = poseColors[pose] || '#94a3b8';
    distHtml += `
      <div class="chart-bar">
        <span class="chart-label">${pose}</span>
        <div class="chart-track"><div class="chart-fill" style="width:${width}%;background:${color}">${count}\u5E27</div></div>
        <span class="chart-val">${pct}%</span>
      </div>`;
  }
  distHtml += '</div>';
  document.getElementById('videoDist').innerHTML = distHtml;

  const smDiv = document.getElementById('saveMoments');
  if (data.save_moments.length > 0) {
    smDiv.innerHTML = '<h3 style="font-size:13px;color:#64748b;margin-bottom:8px">&#128680; 检测到的扑救时刻</h3>' +
      data.save_moments.map(m => `<span class="save-tag">${m.time.toFixed(1)}s</span>`).join('');
  } else {
    smDiv.innerHTML = '<span class="save-tag green">未检测到明显扑救动作</span>';
  }

  const tl = document.getElementById('videoTimeline');
  const dirColors = {'左侧':'#fb923c','正中':'#34d399','右侧':'#38bdf8','未知':'#64748b'};
  tl.innerHTML = '<h3 style="font-size:13px;color:#64748b;margin-bottom:8px;position:sticky;top:0;background:#1e293b;padding:4px 0">&#128336; 逐帧姿态时间线</h3>' +
    data.frames.map(f => {
      const poseCls = f.pose.includes('倒地') ? 'color:#f87171;font-weight:600' : f.pose.includes('侧扑') ? 'color:#fb923c' : '';
      return `<div class="timeline-row">
        <span class="t-time">${f.time.toFixed(1)}s</span>
        <span class="t-pose" style="${poseCls}">${f.pose}</span>
        <span class="t-dir" style="color:${dirColors[f.direction] || '#64748b'}">${f.direction}</span>
        <span class="t-angle">${f.angle}\u00B0</span>
      </div>`;
    }).join('');
}
</script>
</body>
</html>
'''


if __name__ == '__main__':
    print("=" * 60)
    print("     守门员扑救姿态分析系统")
    print("=" * 60)
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"[OK] 模型路径: {MODEL_PATH}")
    else:
        print("[WARNING] 未自动找到模型文件!")
        print("          请手动修改代码中的 MODEL_PATH 变量")
    print("-" * 60)
    print("启动步骤:")
    print("  1. 确保已安装依赖:  pip install ultralytics opencv-python flask")
    print("  2. 在浏览器打开:    http://127.0.0.1:5000")
    print("-" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
