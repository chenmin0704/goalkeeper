import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO


class GoalkeeperVisualizer:
    """
    守门员姿态预测可视化器（适配9关键点版本）
    """

    def __init__(self, model_path, dive_classifier_path=None):
        # 加载YOLO姿态模型
        self.pose_model = YOLO(model_path)
        self.dive_classifier = None

        # 可选：加载扑救方向分类器
        if dive_classifier_path:
            import joblib
            self.dive_classifier = joblib.load(dive_classifier_path)

        # 方向标签和颜色
        self.direction_config = {
            'left': {
                'label': 'LEFT DIVE',
                'color': (255, 128, 0),  # 橙色（左扑）
                'arrow': '←',
            },
            'right': {
                'label': 'RIGHT DIVE',
                'color': (0, 128, 255),  # 蓝色（右扑）
                'arrow': '→',
            },
            'center': {
                'label': 'CENTER',
                'color': (0, 255, 0),  # 绿色（中立）
                'arrow': '↑',
            }
        }

        # ========== 9关键点配置（根据您的实际标注） ==========
        # 0: head
        # 1: shoulde-left
        # 2: shoudle-right
        # 3: hand-left
        # 4: hand-right
        # 5: knee-left
        # 6: knee-right
        # 7: foot-left
        # 8: foot-right

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

        # 骨骼连接（9点版本的简化连接）
        self.skeleton = [
            [0, 1], [0, 2],  # 头到左右肩
            [1, 3], [2, 4],  # 肩到手
            [1, 5], [2, 6],  # 肩到膝
            [5, 7], [6, 8],  # 膝到脚
        ]

    def predict_dive_direction(self, keypoints):
        """
        基于关键点预测扑救方向
        keypoints: [N, 3] 数组 [x, y, visibility]
        """
        if self.dive_classifier:
            return self._ml_predict(keypoints)
        else:
            return self._rule_predict(keypoints)

    def _rule_predict(self, keypoints):
        """
        基于规则预测扑救方向（适配9关键点）
        """
        # 关键点索引（9点版本）
        HEAD = 0
        LEFT_SHOULDER = 1
        RIGHT_SHOULDER = 2
        LEFT_HAND = 3
        RIGHT_HAND = 4
        LEFT_KNEE = 5
        RIGHT_KNEE = 6
        LEFT_FOOT = 7
        RIGHT_FOOT = 8

        score = 0
        reasons = []

        # 检查关键点数量
        if len(keypoints) < 9:
            print(f"警告: 关键点数量不足 ({len(keypoints)}/9)")
            return 'center', 0, ['insufficient_keypoints']

        # 1. 身体倾斜（肩膀中心 vs 头部）
        if keypoints[LEFT_SHOULDER][2] > 0 and keypoints[RIGHT_SHOULDER][2] > 0:
            shoulder_center_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
            head_x = keypoints[HEAD][0]
            head_offset = head_x - shoulder_center_x

            if head_offset < -15:  # 头部明显左偏
                score -= 2
                reasons.append("head_left")
            elif head_offset > 15:  # 头部明显右偏
                score += 2
                reasons.append("head_right")

        # 2. 双手位置
        if keypoints[LEFT_HAND][2] > 0 and keypoints[RIGHT_HAND][2] > 0:
            hand_diff_x = keypoints[RIGHT_HAND][0] - keypoints[LEFT_HAND][0]
            hand_diff_y = keypoints[RIGHT_HAND][1] - keypoints[LEFT_HAND][1]

            # 哪只手更低（y值越大越低，图像坐标系y向下）
            if hand_diff_y > 20:  # 右手更低（倾向左扑）
                score -= 1
                reasons.append("right_hand_low")
            elif hand_diff_y < -20:  # 左手更低（倾向右扑）
                score += 1
                reasons.append("left_hand_low")

            # 哪只手更外侧（相对于肩膀中心）
            if keypoints[LEFT_HAND][0] < shoulder_center_x - 30:
                score -= 1
                reasons.append("left_hand_out")
            if keypoints[RIGHT_HAND][0] > shoulder_center_x + 30:
                score += 1
                reasons.append("right_hand_out")

        # 3. 膝盖位置
        if keypoints[LEFT_KNEE][2] > 0 and keypoints[RIGHT_KNEE][2] > 0:
            knee_diff_y = keypoints[RIGHT_KNEE][1] - keypoints[LEFT_KNEE][1]
            if knee_diff_y > 15:
                score -= 1
                reasons.append("right_knee_low")
            elif knee_diff_y < -15:
                score += 1
                reasons.append("left_knee_low")

        # 4. 脚部位置（支撑脚判断）
        if keypoints[LEFT_FOOT][2] > 0 and keypoints[RIGHT_FOOT][2] > 0:
            foot_diff_x = keypoints[RIGHT_FOOT][0] - keypoints[LEFT_FOOT][0]
            if foot_diff_x > 20:  # 右脚在前/右
                score += 1
                reasons.append("right_foot_forward")
            elif foot_diff_x < -20:  # 左脚在前/左
                score -= 1
                reasons.append("left_foot_forward")

        # 判断方向
        if score <= -2:
            return 'left', score, reasons
        elif score >= 2:
            return 'right', score, reasons
        else:
            return 'center', score, reasons

    def _ml_predict(self, keypoints):
        """使用训练好的分类器预测"""
        features = self._extract_features(keypoints)
        pred = self.dive_classifier.predict([features])[0]
        proba = self.dive_classifier.predict_proba([features])[0]
        return pred, 0, dict(zip(self.dive_classifier.classes_, proba))

    def _extract_features(self, keypoints):
        """提取特征向量（9点版本）"""
        # 简化的特征提取
        features = []

        # 归一化（使用肩膀宽度作为基准）
        if keypoints[1][2] > 0 and keypoints[2][2] > 0:
            shoulder_width = abs(keypoints[2][0] - keypoints[1][0])
            scale = shoulder_width if shoulder_width > 0 else 100
        else:
            scale = 100

        # 身体倾斜
        if keypoints[1][2] > 0 and keypoints[2][2] > 0:
            shoulder_center = (keypoints[1][0] + keypoints[2][0]) / 2
            features.append((keypoints[0][0] - shoulder_center) / scale if keypoints[0][2] > 0 else 0)
        else:
            features.append(0)

        # 手的高度差
        if keypoints[3][2] > 0 and keypoints[4][2] > 0:
            features.append((keypoints[4][1] - keypoints[3][1]) / scale)
        else:
            features.append(0)

        # 手的水平差
        if keypoints[3][2] > 0 and keypoints[4][2] > 0:
            features.append((keypoints[4][0] - keypoints[3][0]) / scale)
        else:
            features.append(0)

        return np.array(features)

    def draw_prediction_on_image(self, image, bbox, keypoints, direction_info):
        """
        在图像上绘制预测结果
        """
        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # 解析方向信息
        if isinstance(direction_info, tuple):
            direction, score, reasons = direction_info
        else:
            direction = direction_info
            score = 0
            reasons = []

        config = self.direction_config[direction]

        # ========== 1. 绘制增强边界框 ==========
        # 外框（粗，方向颜色）
        cv2.rectangle(image, (x1, y1), (x2, y2), config['color'], 3)
        # 内框（细，白色）
        cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

        # ========== 2. 绘制顶部标签栏 ==========
        label = f"{config['arrow']} {config['label']}"

        # 计算文字尺寸
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # 标签背景位置
        label_bg_x1 = x1
        label_bg_y1 = max(y1 - label_h - 15, 0)
        label_bg_x2 = min(x1 + label_w + 20, img_w)
        label_bg_y2 = y1

        # 绘制标签背景
        cv2.rectangle(image,
                      (label_bg_x1, label_bg_y1),
                      (label_bg_x2, label_bg_y2),
                      config['color'], -1)
        cv2.rectangle(image,
                      (label_bg_x1, label_bg_y1),
                      (label_bg_x2, label_bg_y2),
                      (255, 255, 255), 1)

        # 绘制文字
        cv2.putText(image, label,
                    (label_bg_x1 + 10, label_bg_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ========== 3. 绘制侧边置信度条 ==========
        if isinstance(score, (int, float)):
            bar_x = min(x2 + 5, img_w - 15)
            bar_y1 = y1
            bar_y2 = y2
            bar_width = 8

            # 背景条
            cv2.rectangle(image,
                          (bar_x, bar_y1),
                          (bar_x + bar_width, bar_y2),
                          (50, 50, 50), -1)

            # 填充条
            score_norm = min(abs(score) / 5, 1.0)
            fill_height = int((bar_y2 - bar_y1) * score_norm)

            fill_color = config['color']
            if direction == 'center':
                fill_height = int((bar_y2 - bar_y1) * 0.3)  # 中立时显示固定高度

            if direction == 'left':
                # 从下往上填充（左扑）
                cv2.rectangle(image,
                              (bar_x, bar_y2 - fill_height),
                              (bar_x + bar_width, bar_y2),
                              fill_color, -1)
            else:
                # 从上往下填充（右扑/中立）
                cv2.rectangle(image,
                              (bar_x, bar_y1),
                              (bar_x + bar_width, bar_y1 + fill_height),
                              fill_color, -1)

        # ========== 4. 绘制关键点和骨骼 ==========
        self._draw_pose(image, keypoints)

        # ========== 5. 绘制底部信息栏 ==========
        info_y = min(y2 + 25, img_h - 10)

        # 可见关键点数量
        visible_kps = sum(1 for kp in keypoints if kp[2] > 0)
        info_text = f"KPs: {visible_kps}/9 | Score: {score:.1f}"

        (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(image,
                      (x1, info_y - info_h - 5),
                      (min(x1 + info_w + 15, img_w), info_y + 5),
                      (0, 0, 0), -1)
        cv2.putText(image, info_text,
                    (x1 + 5, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def _draw_pose(self, image, keypoints):
        """绘制姿态（关键点和骨骼线）"""
        h, w = image.shape[:2]

        # 关键点颜色映射
        kp_colors = {
            0: (0, 0, 255),  # head - 红色
            1: (0, 255, 0),  # left shoulder - 绿色
            2: (0, 255, 0),  # right shoulder - 绿色
            3: (0, 255, 255),  # left hand - 青色
            4: (0, 255, 255),  # right hand - 青色
            5: (255, 0, 0),  # left knee - 蓝色
            6: (255, 0, 0),  # right knee - 蓝色
            7: (255, 0, 255),  # left foot - 紫色
            8: (255, 0, 255),  # right foot - 紫色
        }

        # 绘制骨骼线
        for connection in self.skeleton:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                kp1 = keypoints[start_idx]
                kp2 = keypoints[end_idx]

                if kp1[2] > 0 and kp2[2] > 0:  # 两个点都可见
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(image, pt1, pt2, (200, 200, 200), 2)

        # 绘制关键点
        for i, (x, y, vis) in enumerate(keypoints):
            if vis > 0 and i in kp_colors:
                color = kp_colors[i]
                # 外圈（白色）
                cv2.circle(image, (int(x), int(y)), 8, (255, 255, 255), 2)
                # 内圈（彩色）
                cv2.circle(image, (int(x), int(y)), 6, color, -1)

                # 小标签
                label = self.kp_names[i].split('-')[0][0].upper()  # 首字母
                if 'left' in self.kp_names[i]:
                    label += 'L'
                elif 'right' in self.kp_names[i]:
                    label += 'R'

                cv2.putText(image, label,
                            (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def process_image(self, image_path, output_path=None):
        """
        处理单张图像：检测 + 预测 + 可视化
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None

        # YOLO检测
        results = self.pose_model(image, verbose=False)

        # 处理每个检测到的目标
        for result in results:
            if result.keypoints is None:
                continue

            # 获取边界框和关键点
            boxes = result.boxes.xyxy.cpu().numpy()

            # 关键点数据
            if result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.cpu().numpy()
                keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

                for i, bbox in enumerate(boxes):
                    # 获取当前目标的关键点
                    if i < len(keypoints_xy):
                        kps_xy = keypoints_xy[i]

                        # 构建 [x, y, visibility] 格式
                        if keypoints_conf is not None and i < len(keypoints_conf):
                            kps_conf = keypoints_conf[i]
                            kp_array = np.array([[kps_xy[j][0], kps_xy[j][1], float(kps_conf[j])]
                                                 for j in range(len(kps_xy))])
                        else:
                            kp_array = np.array([[kps_xy[j][0], kps_xy[j][1], 1.0]
                                                 for j in range(len(kps_xy))])

                        # 预测扑救方向
                        direction_info = self.predict_dive_direction(kp_array)

                        # 绘制到图像上
                        image = self.draw_prediction_on_image(image, bbox, kp_array, direction_info)

        # 保存或显示
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"结果已保存: {output_path}")

        return image

    def process_video(self, video_path, output_path=None, display=True):
        """
        处理视频流
        """
        cap = cv2.VideoCapture(video_path)

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 每2帧处理一次（平衡速度和流畅度）
            if frame_count % 2 == 0:
                results = self.pose_model(frame, verbose=False)

                for result in results:
                    if result.keypoints is None:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()
                    keypoints_xy = result.keypoints.xy.cpu().numpy()
                    keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

                    for i, bbox in enumerate(boxes):
                        if i < len(keypoints_xy):
                            kps_xy = keypoints_xy[i]

                            if keypoints_conf is not None and i < len(keypoints_conf):
                                kps_conf = keypoints_conf[i]
                                kp_array = np.array([[kps_xy[j][0], kps_xy[j][1], float(kps_conf[j])]
                                                     for j in range(len(kps_xy))])
                            else:
                                kp_array = np.array([[kps_xy[j][0], kps_xy[j][1], 1.0]
                                                     for j in range(len(kps_xy))])

                            direction_info = self.predict_dive_direction(kp_array)
                            frame = self.draw_prediction_on_image(frame, bbox, kp_array, direction_info)

            # 显示
            if display:
                cv2.imshow('Goalkeeper Dive Prediction (Press Q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 保存
            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"处理了 {frame_count} 帧")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 配置路径（请修改为您的实际路径）
    YOLO_MODEL = r"C:\Users\chen3\Desktop\ok\goalkeeper_training\exp_cpu\weights\best.pt"
    IMAGE_PATH = r"C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames\frame_00001.jpg"
    OUTPUT_DIR = r"C:\Users\chen3\Desktop\ok\predictions"

    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # 初始化可视化器
    visualizer = GoalkeeperVisualizer(YOLO_MODEL)

    # 处理单张图像
    output_path = Path(OUTPUT_DIR) / "prediction_result.jpg"
    result = visualizer.process_image(IMAGE_PATH, str(output_path))

    # 或者处理视频
    # VIDEO_PATH = r"C:\Users\chen3\Desktop\ok\pujiu5.mp4"
    # visualizer.process_video(VIDEO_PATH, str(Path(OUTPUT_DIR) / "output.mp4"))

    print("完成！")