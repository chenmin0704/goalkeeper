
import torch
import pickle
import numpy as np
import cv2
from ultralytics import YOLO
import json
import os
import warnings

class GoalkeeperDivePredictor:
    """
    守门员扑救方向预测器
    """
    def __init__(self, pose_model_path, dive_model_dir):
        # 检查姿态模型是否存在
        if not os.path.exists(pose_model_path):
            print(f"⚠️ 姿态模型不存在: {pose_model_path}")
            print("将尝试使用预训练模型 yolov8n-pose.pt")
            print("注意：预训练模型可能不适用于你的特定场景\n")
            pose_model_path = 'yolov8n-pose.pt'

        # 加载YOLO姿态估计模型
        try:
            self.pose_model = YOLO(pose_model_path)
            print(f"✓ 姿态模型加载成功: {pose_model_path}")
        except Exception as e:
            print(f"✗ 姿态模型加载失败: {e}")
            raise

        # 加载方向预测模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 检查并加载数据集配置
        dataset_path = f'{dive_model_dir}/dataset.pkl'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        # 动态导入模型类（避免循环导入）
        self.dive_model = self._create_model(data['feature_dim'])

        # 加载模型权重
        model_path = f'{dive_model_dir}/best_model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重不存在: {model_path}")

        try:
            # 修复：添加 weights_only=True
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.dive_model.load_state_dict(state_dict)
            self.dive_model.eval()
            print(f"✓ 方向预测模型加载成功")
        except Exception as e:
            print(f"⚠️ 使用兼容模式加载: {e}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.dive_model.load_state_dict(state_dict)
            self.dive_model.eval()

        self.scaler = data['scaler']

        # 关键点映射
        self.kp_names = [
            'head', 'shoulde-left', 'shoudle-right',
            'hand-left', 'hand-right',
            'knee-left', 'knee-right',
            'foot-left', 'foot-right'
        ]

        print("模型加载完成！\n")

    def _create_model(self, input_dim, hidden_dim=64):
        """创建神经网络模型"""
        import torch.nn as nn
        class DiveDirectionNet(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 3)
                )
            def forward(self, x):
                return self.net(x)

        return DiveDirectionNet(input_dim, hidden_dim).to(self.device)

    def extract_features_from_pose(self, keypoints, img_width, img_height):
        """从YOLO姿态结果提取特征"""
        kp_dict = {}
        for i, name in enumerate(self.kp_names):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                if conf > 0.3:
                    kp_dict[name] = {'x': x * img_width, 'y': y * img_height}

        features = self._compute_features(kp_dict, img_width, img_height)
        return features, kp_dict

    def _compute_features(self, kps, w, h):
        """计算特征向量"""
        features = []

        # 归一化坐标 (18维)
        coords = []
        for name in self.kp_names:
            if name in kps:
                coords.extend([kps[name]['x'] / w, kps[name]['y'] / h])
            else:
                coords.extend([0, 0])
        features.extend(coords)

        def get(name):
            return (kps[name]['x'] / w, kps[name]['y'] / h) if name in kps else None

        # 头部偏移 (2维)
        head = get('head')
        sh_l = get('shoulde-left')
        sh_r = get('shoudle-right')
        if head and sh_l and sh_r:
            features.extend([head[0] - (sh_l[0]+sh_r[0])/2, head[1] - (sh_l[1]+sh_r[1])/2])
        else:
            features.extend([0, 0])

        # 手部高度 (3维)
        hand_l = get('hand-left')
        hand_r = get('hand-right')
        if hand_l and hand_r and sh_l and sh_r:
            features.extend([hand_l[1]-sh_l[1], hand_r[1]-sh_r[1],
                           (hand_l[1]-sh_l[1]) - (hand_r[1]-sh_r[1])])
        else:
            features.extend([0, 0, 0])

        # 膝盖弯曲 (4维)
        knee_l = get('knee-left')
        knee_r = get('knee-right')
        foot_l = get('foot-left')
        foot_r = get('foot-right')
        if knee_l and knee_r and foot_l and foot_r:
            features.extend([knee_l[1]-foot_l[1], knee_r[1]-foot_r[1],
                          knee_l[0]-foot_l[0], knee_r[0]-foot_r[0]])
        else:
            features.extend([0, 0, 0, 0])

        # 重心和跨度 (2维)
        valid_x = [c for i, c in enumerate(coords) if i % 2 == 0 and c > 0]
        if valid_x:
            features.append(np.mean(valid_x) - 0.5)
            features.append(max(valid_x) - min(valid_x))
        else:
            features.extend([0, 0])

        return np.array(features, dtype=np.float32)

    def predict(self, image_path, conf_threshold=0.3):
        """预测单张图片的扑救方向"""
        # YOLO姿态估计
        results = self.pose_model(image_path, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return None, "未检测到守门员"

        img = cv2.imread(image_path)
        if img is None:
            return None, f"无法读取图片: {image_path}"

        h, w = img.shape[:2]
        predictions = []

        for i, kp_data in enumerate(results.keypoints.data):
            keypoints = kp_data.cpu().numpy()

            # 提取特征
            features, kp_dict = self.extract_features_from_pose(keypoints, w, h)

            # 标准化并预测
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)

            with torch.no_grad():
                output = self.dive_model(features_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = torch.argmax(output).item()

            direction = pred_class - 1
            direction_names = {-1: 'LEFT', 0: 'CENTER', 1: 'RIGHT'}

            predictions.append({
                'person_id': i,
                'direction': direction,
                'direction_name': direction_names[direction],
                'confidence': float(probs[pred_class]),
                'probabilities': {
                    'left': float(probs[0]),
                    'center': float(probs[1]),
                    'right': float(probs[2])
                },
                'keypoints': kp_dict
            })

        return predictions, img

    def visualize_prediction(self, image_path, output_path=None):
        """可视化预测结果"""
        predictions, img = self.predict(image_path)

        if predictions is None:
            print("未检测到守门员")
            return None, None

        for pred in predictions:
            kp = pred['keypoints']
            direction = pred['direction_name']
            conf = pred['confidence']

            # 绘制关键点
            color_map = {
                'head': (0, 0, 255),
                'shoulde-left': (0, 255, 0), 'shoudle-right': (0, 255, 0),
                'hand-left': (0, 255, 255), 'hand-right': (0, 255, 255),
                'knee-left': (255, 0, 0), 'knee-right': (255, 0, 0),
                'foot-left': (255, 0, 255), 'foot-right': (255, 0, 255)
            }

            for name, coord in kp.items():
                x, y = int(coord['x']), int(coord['y'])
                color = color_map.get(name, (128, 128, 128))
                cv2.circle(img, (x, y), 6, color, -1)

            # 绘制骨骼
            skeleton = [
                ('head', 'shoulde-left'), ('head', 'shoudle-right'),
                ('shoulde-left', 'hand-left'), ('shoudle-right', 'hand-right'),
                ('shoulde-left', 'knee-left'), ('shoudle-right', 'knee-right'),
                ('knee-left', 'foot-left'), ('knee-right', 'foot-right')
            ]
            for start, end in skeleton:
                if start in kp and end in kp:
                    x1, y1 = int(kp[start]['x']), int(kp[start]['y'])
                    x2, y2 = int(kp[end]['x']), int(kp[end]['y'])
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 显示预测结果
            text = f"{direction} ({conf:.2f})"
            text_colors = {'LEFT': (255, 0, 0), 'CENTER': (0, 255, 0), 'RIGHT': (0, 0, 255)}
            text_color = text_colors.get(direction, (255, 255, 255))

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
            cv2.putText(img, text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            # 概率条
            bar_y = 50
            for j, (name, prob) in enumerate(pred['probabilities'].items()):
                bar_width = int(prob * 200)
                y_pos = bar_y + j * 25
                cv2.rectangle(img, (10, y_pos), (10 + bar_width, y_pos + 20), text_color, -1)
                cv2.putText(img, f"{name}: {prob:.2f}", (220, y_pos + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"✓ 结果保存至: {output_path}")

        return img, predictions

if __name__ == '__main__':
    # 路径配置（根据实际情况修改）

    # 方案1: 使用你训练的姿态模型（如果存在）
    pose_model = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\goalkeeper_training\exp_cpu\weights\best.pt'

    # 方案2: 如果上述不存在，自动回退到预训练模型
    if not os.path.exists(pose_model):
        pose_model = 'yolov8n-pose.pt'  # 自动下载

    dive_model_dir = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction'

    print("="*60)
    print("守门员扑救方向预测系统")
    print("="*60)

    # 初始化
    try:
        predictor = GoalkeeperDivePredictor(pose_model, dive_model_dir)
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        print("\n请检查:")
        print("1. 是否已完成方向预测模型训练 (train_dive_model.py)")
        print("2. 路径是否正确")
        exit(1)

    # 测试图片
    test_image = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames\frame_00023.jpg'
    output = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction\test_result.jpg'

    print("单张图片预测")
    print("-"*60)

    if not os.path.exists(test_image):
        print(f"✗ 测试图片不存在: {test_image}")
        # 尝试查找其他图片
        import glob
        alt_images = glob.glob(r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames\*.jpg')
        if alt_images:
            test_image = alt_images[0]
            print(f"使用替代图片: {test_image}")
        else:
            print("未找到任何测试图片")
            exit(1)

    # 运行预测
    img, pred = predictor.visualize_prediction(test_image, output)

    if pred:
        print(f"\n预测结果:")
        print(f"  方向: {pred[0]['direction_name']}")
        print(f"  置信度: {pred[0]['confidence']:.3f}")
        print(f"  概率分布:")
        for name, prob in pred[0]['probabilities'].items():
            print(f"    {name}: {prob:.3f}")
    else:
        print("预测失败")
