# 4.visualize.py - 预测可视化（9关键点版本）
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class GoalkeeperVisualizer:
    """守门员姿态预测可视化器（9关键点版本）"""

    def __init__(self, model_path):
        self.pose_model = YOLO(model_path)

        # 9关键点配置
        self.kp_names = {
            0: 'head',
            1: 'shoulde-left',
            2: 'shoudle-right',
            3: 'hand-left',
            4: 'hand-right',
            5: 'knee-left',
            6: 'knee-right',
            7: 'foot-left',
            8: 'foot-right'
        }

        # 骨骼连接
        self.skeleton = [
            [0, 1], [0, 2],  # 头到肩
            [1, 3], [2, 4],  # 肩到手
            [1, 5], [2, 6],  # 肩到膝
            [5, 7], [6, 8],  # 膝到脚
        ]

        # 方向配置
        self.direction_config = {
            'left': {'label': '← LEFT DIVE', 'color': (255, 128, 0)},
            'right': {'label': '→ RIGHT DIVE', 'color': (0, 128, 255)},
            'center': {'label': '↑ CENTER', 'color': (0, 255, 0)}
        }

    def predict_dive_direction(self, keypoints):
        """基于规则预测扑救方向"""
        if len(keypoints) < 9:
            return 'center', 0, ['insufficient_keypoints']

        # 索引
        HEAD, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1, 2
        LEFT_HAND, RIGHT_HAND = 3, 4
        LEFT_KNEE, RIGHT_KNEE = 5, 6

        score = 0
        reasons = []

        # 1. 头部偏移
        if keypoints[LEFT_SHOULDER][2] > 0 and keypoints[RIGHT_SHOULDER][2] > 0:
            shoulder_center_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
            head_offset = keypoints[HEAD][0] - shoulder_center_x

            if head_offset < -15:
                score -= 2
                reasons.append("head_left")
            elif head_offset > 15:
                score += 2
                reasons.append("head_right")

        # 2. 双手位置
        if keypoints[LEFT_HAND][2] > 0 and keypoints[RIGHT_HAND][2] > 0:
            hand_diff_y = keypoints[RIGHT_HAND][1] - keypoints[LEFT_HAND][1]

            if hand_diff_y > 20:
                score -= 1
                reasons.append("right_hand_low")
            elif hand_diff_y < -20:
                score += 1
                reasons.append("left_hand_low")

        # 3. 膝盖位置
        if keypoints[LEFT_KNEE][2] > 0 and keypoints[RIGHT_KNEE][2] > 0:
            knee_diff_y = keypoints[RIGHT_KNEE][1] - keypoints[LEFT_KNEE][1]
            if knee_diff_y > 15:
                score -= 1
                reasons.append("right_knee_low")
            elif knee_diff_y < -15:
                score += 1
                reasons.append("left_knee_low")

        # 判断
        if score <= -2:
            return 'left', score, reasons
        elif score >= 2:
            return 'right', score, reasons
        else:
            return 'center', score, reasons

    def draw_prediction(self, image, bbox, keypoints, direction_info):
        """在图像上绘制预测结果"""
        x1, y1, x2, y2 = map(int, bbox)
        direction, score, reasons = direction_info
        config = self.direction_config[direction]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), config['color'], 3)
        cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

        # 绘制标签
        label = config['label']
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        label_bg_y1 = max(y1 - label_h - 15, 0)
        cv2.rectangle(image, (x1, label_bg_y1), (x1 + label_w + 20, y1), config['color'], -1)
        cv2.putText(image, label, (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 绘制骨骼
        for connection in self.skeleton:
            i1, i2 = connection
            if i1 < len(keypoints) and i2 < len(keypoints):
                kp1, kp2 = keypoints[i1], keypoints[i2]
                if kp1[2] > 0 and kp2[2] > 0:
                    cv2.line(image, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (200, 200, 200), 2)

        # 绘制关键点
        for i, (x, y, vis) in enumerate(keypoints):
            if vis > 0:
                color = (0, 255, 0) if i < 5 else ((255, 0, 0) if i < 7 else (255, 0, 255))
                cv2.circle(image, (int(x), int(y)), 6, color, -1)
                cv2.circle(image, (int(x), int(y)), 8, (255, 255, 255), 2)

        # 底部信息
        visible_kps = sum(1 for kp in keypoints if kp[2] > 0)
        info_text = f"KPs: {visible_kps}/9 | Score: {score:.1f}"
        cv2.putText(image, info_text, (x1, min(y2 + 20, image.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def process_image(self, image_path, output_path=None):
        """处理单张图像"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取: {image_path}")
            return None

        results = self.pose_model(image, verbose=False)

        for result in results:
            if result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            confs = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            for i, bbox in enumerate(boxes):
                if i < len(keypoints_xy):
                    kps = keypoints_xy[i]
                    conf = confs[i] if confs is not None else [1.0] * len(kps)

                    kp_array = np.array([[kps[j][0], kps[j][1], float(conf[j])] for j in range(len(kps))])
                    direction_info = self.predict_dive_direction(kp_array)
                    image = self.draw_prediction(image, bbox, kp_array, direction_info)

        if output_path:
            cv2.imwrite(output_path, image)
            print(f"保存: {output_path}")

        return image


if __name__ == '__main__':
    MODEL = r"C:\Users\chen3\Desktop\ok\goalkeeper_training\exp_cpu\weights\best.pt"
    IMAGE = r"C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames\frame_00001.jpg"
    OUTPUT = r"C:\Users\chen3\Desktop\ok\predictions\result.jpg"

    Path(OUTPUT).parent.mkdir(exist_ok=True)

    visualizer = GoalkeeperVisualizer(MODEL)
    visualizer.process_image(IMAGE, OUTPUT)