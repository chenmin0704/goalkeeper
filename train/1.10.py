#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO


class GoalkeeperVisualizer:
    """守门员姿态预测可视化器（9关键点版本）"""

    def __init__(self, model_path):
        self.pose_model = YOLO(model_path)

        self.kp_names = {
            0: 'head', 1: 'shoulde-left', 2: 'shoudle-right',
            3: 'hand-left', 4: 'hand-right', 5: 'knee-left',
            6: 'knee-right', 7: 'foot-left', 8: 'foot-right'
        }

        self.skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],
            [1, 5], [2, 6], [5, 7], [6, 8],
        ]

        self.direction_config = {
            'left': {'label': '← LEFT DIVE', 'color': (255, 128, 0)},
            'right': {'label': '→ RIGHT DIVE', 'color': (0, 128, 255)},
            'center': {'label': '↑ CENTER', 'color': (0, 255, 0)}
        }

    def predict_dive_direction(self, keypoints, head_threshold=15,
                               hand_threshold=20, knee_threshold=15,
                               score_threshold=2):
        """基于规则预测扑救方向"""
        if len(keypoints) < 9:
            return 'center', 0, ['insufficient_keypoints']

        HEAD, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1, 2
        LEFT_HAND, RIGHT_HAND = 3, 4
        LEFT_KNEE, RIGHT_KNEE = 5, 6

        score = 0
        reasons = []

        if keypoints[LEFT_SHOULDER][2] > 0 and keypoints[RIGHT_SHOULDER][2] > 0:
            shoulder_center_x = (keypoints[LEFT_SHOULDER][0] +
                                 keypoints[RIGHT_SHOULDER][0]) / 2
            head_offset = keypoints[HEAD][0] - shoulder_center_x

            if head_offset < -head_threshold:
                score -= 2
                reasons.append("head_left")
            elif head_offset > head_threshold:
                score += 2
                reasons.append("head_right")

        if keypoints[LEFT_HAND][2] > 0 and keypoints[RIGHT_HAND][2] > 0:
            hand_diff_y = keypoints[RIGHT_HAND][1] - keypoints[LEFT_HAND][1]
            if hand_diff_y > hand_threshold:
                score -= 1
                reasons.append("right_hand_low")
            elif hand_diff_y < -hand_threshold:
                score += 1
                reasons.append("left_hand_low")

        if keypoints[LEFT_KNEE][2] > 0 and keypoints[RIGHT_KNEE][2] > 0:
            knee_diff_y = keypoints[RIGHT_KNEE][1] - keypoints[LEFT_KNEE][1]
            if knee_diff_y > knee_threshold:
                score -= 1
                reasons.append("right_knee_low")
            elif knee_diff_y < -knee_threshold:
                score += 1
                reasons.append("left_knee_low")

        if score <= -score_threshold:
            return 'left', score, reasons
        elif score >= score_threshold:
            return 'right', score, reasons
        else:
            return 'center', score, reasons

    def draw_prediction(self, image, bbox, keypoints, direction_info):
        """在图像上绘制预测结果"""
        x1, y1, x2, y2 = map(int, bbox)
        direction, score, reasons = direction_info
        config = self.direction_config[direction]

        cv2.rectangle(image, (x1, y1), (x2, y2), config['color'], 3)
        cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                      (255, 255, 255), 1)

        label = config['label']
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        label_bg_y1 = max(y1 - label_h - 15, 0)
        cv2.rectangle(image, (x1, label_bg_y1),
                      (x1 + label_w + 20, y1), config['color'], -1)
        cv2.putText(image, label, (x1 + 10, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        for connection in self.skeleton:
            i1, i2 = connection
            if i1 < len(keypoints) and i2 < len(keypoints):
                kp1, kp2 = keypoints[i1], keypoints[i2]
                if kp1[2] > 0 and kp2[2] > 0:
                    cv2.line(image,
                             (int(kp1[0]), int(kp1[1])),
                             (int(kp2[0]), int(kp2[1])),
                             (200, 200, 200), 2)

        for i, (x, y, vis) in enumerate(keypoints):
            if vis > 0:
                color = (0, 255, 0) if i < 5 else (
                    (255, 0, 0) if i < 7 else (255, 0, 255)
                )
                cv2.circle(image, (int(x), int(y)), 6, color, -1)
                cv2.circle(image, (int(x), int(y)), 8, (255, 255, 255), 2)

        visible_kps = sum(1 for kp in keypoints if kp[2] > 0)
        info_text = f"KPs: {visible_kps}/9 | Score: {score:.1f}"
        cv2.putText(image, info_text,
                    (x1, min(y2 + 20, image.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image


# ==================== Gradio Web应用 ====================

MODEL_PATH = r"C:\Users\chen3\Desktop\ok\goalkeeper_training\exp_cpu\weights\best.pt"
visualizer = GoalkeeperVisualizer(MODEL_PATH)

TITLE = "基于YOLOv8的守门员扑救预测系统"
DESCRIPTION = """
上传守门员扑救图像，系统自动检测姿态并预测扑救方向（左/中/右）。
"""


def predict_image(img, conf_threshold, head_threshold,
                  hand_threshold, knee_threshold, score_threshold):
    """预测图像中的守门员姿态并判断扑救方向"""
    if img is None:
        return None

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    results = visualizer.pose_model.predict(
        source=img_cv,
        conf=conf_threshold,
        verbose=False
    )

    for result in results:
        if result.keypoints is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        for i, bbox in enumerate(boxes):
            if i >= len(keypoints_xy):
                continue

            kps = keypoints_xy[i]
            conf = confs[i] if confs is not None else [1.0] * len(kps)

            kp_array = np.array([
                [float(kps[j][0]), float(kps[j][1]), float(conf[j])]
                for j in range(len(kps))
            ])

            direction_info = visualizer.predict_dive_direction(
                kp_array,
                head_threshold=int(head_threshold),
                hand_threshold=int(hand_threshold),
                knee_threshold=int(knee_threshold),
                score_threshold=int(score_threshold)
            )

            img_cv = visualizer.draw_prediction(img_cv, bbox, kp_array, direction_info)

    result_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return result_img


# 创建Gradio界面（Gradio 6.0兼容）
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="上传守门员图像"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                  label="检测置信度阈值"),
        gr.Slider(minimum=5, maximum=30, value=15, step=1,
                  label="头部偏移阈值（像素）"),
        gr.Slider(minimum=10, maximum=40, value=20, step=1,
                  label="双手高度差阈值（像素）"),
        gr.Slider(minimum=10, maximum=30, value=15, step=1,
                  label="双膝高度差阈值（像素）"),
        gr.Slider(minimum=1, maximum=4, value=2, step=1,
                  label="方向判定总分阈值"),
    ],
    outputs=gr.Image(type="pil", label="扑救方向预测结果"),
    title=TITLE,
    description=DESCRIPTION,
    flagging_mode="never",
)

if __name__ == "__main__":
    # Gradio 6.0: theme参数移至launch()
    iface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,  # 更换为7861端口
        show_error=True,
        theme=gr.themes.Soft(),  # 移至此处
    )