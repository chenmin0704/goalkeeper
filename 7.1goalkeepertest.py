"""
守门员扑救动作预测分析
基于YOLOv8-Pose关键点检测结果
"""

import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque

# ==================== 配置 ====================
MODEL_PATH = r"C:\Users\chen3\Desktop\ok\新建文件夹\runs\goalkeeper_pose\weights\best.pt"
CONF_THRESHOLD = 0.3  # 置信度阈值
# =============================================

# 关键点索引（对应YOLOv8-Pose输出顺序）
KP_IDX = {
    'head': 0,
    'shoulder_left': 1,
    'shoulder_right': 2,
    'hand_left': 3,
    'hand_right': 4,
    'knee_left': 5,
    'knee_right': 6,
    'foot_left': 7,
    'foot_right': 8
}

# 骨骼连接（用于可视化）
SKELETON = [
    (0, 1), (0, 2),  # head -> shoulders
    (1, 3), (2, 4),  # shoulders -> hands
    (1, 5), (2, 6),  # shoulders -> knees
    (5, 7), (6, 8),  # knees -> feet
    (1, 2),  # shoulder left -> shoulder right
    (5, 6),  # knee left -> knee right
    (7, 8),  # foot left -> foot right
]


def load_model(model_path):
    """加载训练好的YOLOv8-Pose模型"""
    return YOLO(model_path)


def extract_keypoints(result, conf_thresh=CONF_THRESHOLD):
    """
    从YOLOv8预测结果中提取关键点
    返回: [(x, y, conf), ...] 或 None
    """
    if result.keypoints is None:
        return None

    kpts = result.keypoints
    if kpts.xy is None or len(kpts.xy) == 0:
        return None

    # 取第一个检测到的人体
    xy = kpts.xy[0].cpu().numpy()  # (9, 2)
    conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else np.ones(9)

    keypoints = []
    for i in range(9):
        x, y = xy[i]
        c = float(conf[i]) if i < len(conf) else 1.0
        if c < conf_thresh:
            keypoints.append((None, None, c))
        else:
            keypoints.append((int(x), int(y), c))

    return keypoints


def draw_pose(img, keypoints, color=(0, 255, 0)):
    """在图像上绘制关键点和骨骼"""
    if keypoints is None:
        return img

    h, w = img.shape[:2]

    # 绘制骨骼线
    for start_idx, end_idx in SKELETON:
        pt1 = keypoints[start_idx]
        pt2 = keypoints[end_idx]
        if pt1[0] is not None and pt2[0] is not None:
            cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 255, 0), 2)

    # 绘制关键点
    kp_names = ['头', '左肩', '右肩', '左手', '右手', '左膝', '右膝', '左脚', '右脚']
    for i, (x, y, conf) in enumerate(keypoints):
        if x is not None:
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.putText(img, kp_names[i], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def point_distance(p1, p2):
    """计算两点距离"""
    if p1[0] is None or p2[0] is None:
        return None
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_angle(p1, p2):
    """计算两点连线与水平线的夹角（度）"""
    if p1[0] is None or p2[0] is None:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def compute_body_angle(keypoints):
    """
    计算身体躯干角度（头肩中点 -> 膝中点）
    返回值: 与垂直方向的夹角（度），正值表示右倾
    """
    head = keypoints[0]
    ls = keypoints[1]
    rs = keypoints[2]
    lk = keypoints[5]
    rk = keypoints[6]

    # 肩中点
    if ls[0] is not None and rs[0] is not None:
        shoulder_cx = (ls[0] + rs[0]) / 2
        shoulder_cy = (ls[1] + rs[1]) / 2
    elif ls[0] is not None:
        shoulder_cx, shoulder_cy = ls[0], ls[1]
    elif rs[0] is not None:
        shoulder_cx, shoulder_cy = rs[0], rs[1]
    else:
        return None

    # 膝中点
    if lk[0] is not None and rk[0] is not None:
        knee_cx = (lk[0] + rk[0]) / 2
        knee_cy = (lk[1] + rk[1]) / 2
    elif lk[0] is not None:
        knee_cx, knee_cy = lk[0], lk[1]
    elif rk[0] is not None:
        knee_cx, knee_cy = rk[0], rk[1]
    else:
        return None

    dx = knee_cx - shoulder_cx
    dy = knee_cy - shoulder_cy
    angle = math.degrees(math.atan2(dx, abs(dy)))  # 与垂直方向的夹角
    return angle


def classify_pose(keypoints):
    """
    基于关键点分类守门员姿态
    返回: (姿态名称, 置信度, 详细描述)
    """
    if keypoints is None:
        return "未检测", 0.0, "未检测到守门员"

    head = keypoints[0]
    lh = keypoints[3]
    rh = keypoints[4]
    lk = keypoints[5]
    rk = keypoints[6]
    lf = keypoints[7]
    rf = keypoints[8]

    # 检查关键点可用性
    visible_count = sum(1 for kp in keypoints if kp[0] is not None)
    if visible_count < 4:
        return "无法判断", 0.3, "关键点不足"

    # 计算身体角度
    body_angle = compute_body_angle(keypoints)

    # 手的高度位置（相对于头部）
    hand_high = False
    hand_low = False
    hand_spread = False

    if lh[0] is not None and rh[0] is not None:
        hand_cy = (lh[1] + rh[1]) / 2
        if head[0] is not None and hand_cy < head[1] + 20:
            hand_high = True
        if hand_cy > head[1] + 80:
            hand_low = True
        hand_distance = abs(lh[0] - rh[0])
        if hand_distance > 100:
            hand_spread = True

    # 脚的位置关系（判断是否离地/倒地）
    feet_close = False
    if lf[0] is not None and rf[0] is not None:
        foot_dist = point_distance(lf, rf)
        if foot_dist is not None and foot_dist < 50:
            feet_close = True

    # 判断逻辑
    pose = "站立准备"
    confidence = 0.5
    details = []

    # 身体倾斜判断
    if body_angle is not None:
        if abs(body_angle) > 45:
            pose = "倒地扑救"
            confidence = min(abs(body_angle) / 90, 1.0)
            if body_angle > 0:
                details.append(f"向右倒地 ({body_angle:.1f}度)")
            else:
                details.append(f"向左倒地 ({body_angle:.1f}度)")
        elif abs(body_angle) > 20:
            pose = "侧扑"
            confidence = abs(body_angle) / 60
            if body_angle > 0:
                details.append(f"向右侧扑 ({body_angle:.1f}度)")
            else:
                details.append(f"向左侧扑 ({body_angle:.1f}度)")
        else:
            details.append(f"身体直立 ({body_angle:.1f}度)")

    # 手部位置判断
    if hand_high and hand_spread:
        pose = "高空接球" if pose in ["站立准备"] else pose
        details.append("双手高举")
    elif hand_spread:
        if "倒地" in pose or "侧扑" in pose:
            details.append("单手伸展扑救")
        else:
            details.append("双手展开准备")
    elif hand_low:
        details.append("手部低位")

    if feet_close:
        details.append("双脚并拢/离地")

    detail_str = "; ".join(details) if details else "常规姿态"
    return pose, confidence, detail_str


def predict_save_direction(keypoints):
    """
    预测扑救方向
    返回: 方向, 概率
    """
    if keypoints is None:
        return "未知", 0.0

    lh = keypoints[3]
    rh = keypoints[4]
    head = keypoints[0]

    if lh[0] is None or rh[0] is None:
        return "未知", 0.0

    # 计算手的水平中心相对于头部的位置
    hand_cx = (lh[0] + rh[0]) / 2
    if head[0] is not None:
        diff = hand_cx - head[0]
        if diff < -40:
            return "左侧", min(abs(diff) / 100, 1.0)
        elif diff > 40:
            return "右侧", min(abs(diff) / 100, 1.0)

    return "正中", 0.5


def analyze_coverage(keypoints, img_w, img_h):
    """
    分析扑救覆盖范围
    返回: 覆盖区域百分比, 描述
    """
    if keypoints is None:
        return 0.0, "未检测"

    visible_pts = [(kp[0], kp[1]) for kp in keypoints if kp[0] is not None]
    if len(visible_pts) < 3:
        return 0.0, "关键点不足"

    # 计算关键点覆盖的矩形区域
    xs = [p[0] for p in visible_pts]
    ys = [p[1] for p in visible_pts]

    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    img_area = img_w * img_h
    coverage = min(area / img_area * 100, 100)

    if coverage > 15:
        desc = "大幅伸展"
    elif coverage > 8:
        desc = "中等范围"
    else:
        desc = "紧凑姿态"

    return coverage, desc


def predict_on_image(model, image_path, save_path=None):
    """
    对单张图像进行预测分析
    """
    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None

    img_h, img_w = img.shape[:2]

    # YOLOv8推理
    results = model(image_path, conf=CONF_THRESHOLD)

    # 分析结果
    for result in results:
        keypoints = extract_keypoints(result)

        # 绘制姿态
        img = draw_pose(img, keypoints)

        # 分类姿态
        pose, conf, detail = classify_pose(keypoints)

        # 预测方向
        direction, dir_prob = predict_save_direction(keypoints)

        # 覆盖范围
        coverage, cov_desc = analyze_coverage(keypoints, img_w, img_h)

        # 在图像上绘制分析结果
        y_offset = 30
        texts = [
            f"姿态: {pose} ({conf:.0%})",
            f"方向: {direction} ({dir_prob:.0%})",
            f"覆盖: {cov_desc} ({coverage:.1f}%)",
            f"详情: {detail}",
        ]

        for text in texts:
            cv2.putText(img, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

    # 保存或显示
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"结果已保存: {save_path}")

    return {
        'pose': pose,
        'pose_conf': conf,
        'direction': direction,
        'direction_conf': dir_prob,
        'coverage': coverage,
        'coverage_desc': cov_desc,
        'detail': detail,
        'keypoints': keypoints
    }


def predict_on_video(model, video_path, output_path=None):
    """
    对视频进行预测分析
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 动作时序分析
    pose_history = deque(maxlen=fps * 2)  # 保留最近2秒
    action_sequence = []

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None

    frame_idx = 0
    print(f"处理视频: {video_path} ({fps}fps, {w}x{h})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 3 != 0:  # 每3帧处理一次，加速
            if out:
                out.write(frame)
            continue

        # 推理
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        for result in results:
            keypoints = extract_keypoints(result)

            # 绘制
            frame = draw_pose(frame, keypoints)

            # 分析
            pose, conf, detail = classify_pose(keypoints)
            direction, dir_prob = predict_save_direction(keypoints)
            coverage, cov_desc = analyze_coverage(keypoints, w, h)

            pose_history.append(pose)
            action_sequence.append({
                'frame': frame_idx,
                'pose': pose,
                'direction': direction
            })

            # 绘制信息
            y = 30
            infos = [
                f"姿态: {pose}",
                f"方向: {direction}",
                f"覆盖: {cov_desc}",
                f"帧: {frame_idx}"
            ]
            for info in infos:
                cv2.putText(frame, info, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25

        if out:
            out.write(frame)

        if frame_idx % 30 == 0:
            print(f"  已处理 {frame_idx} 帧")

    cap.release()
    if out:
        out.release()

    # 生成时序报告
    print("\n" + "=" * 60)
    print("视频动作分析报告")
    print("=" * 60)

    from collections import Counter
    pose_counts = Counter(a['pose'] for a in action_sequence)
    print("\n姿态分布:")
    for pose, count in pose_counts.most_common():
        print(f"  {pose}: {count}帧 ({count / len(action_sequence) * 100:.1f}%)")

    # 检测扑救动作序列
    save_moments = []
    for i, a in enumerate(action_sequence):
        if a['pose'] in ['倒地扑救', '侧扑', '高空接球']:
            save_moments.append(a)

    print(f"\n检测到 {len(save_moments)} 次扑救动作")

    return action_sequence


def main():
    print("=" * 60)
    print("守门员扑救动作预测分析")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("模型加载完成!")

    # 示例：单张图像预测
    print("\n--- 单张图像预测 ---")
    # predict_on_image(model, "test_image.jpg", "result.jpg")

    # 示例：视频预测
    print("\n--- 视频预测 ---")
    # predict_on_video(model, "test_video.mp4", "result_video.mp4")

    print("\n使用方法:")
    print("  图像预测: predict_on_image(model, '图片路径', '保存路径')")
    print("  视频预测: predict_on_video(model, '视频路径', '保存路径')")


if __name__ == '__main__':
    main()