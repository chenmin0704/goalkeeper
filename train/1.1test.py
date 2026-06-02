# 生成端到端推理代码（结合YOLO姿态估计）
import torch
import pickle
import numpy as np
import cv2
from ultralytics import YOLO
import json

class GoalkeeperDivePredictor:
    """
    守门员扑救方向预测器
    结合YOLO姿态估计和方向分类
    """
    def __init__(self, pose_model_path, dive_model_dir):
        # 加载YOLO姿态估计模型
        self.pose_model = YOLO(pose_model_path)

        # 加载方向预测模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(f'{dive_model_dir}/dataset.pkl', 'rb') as f:
            data = pickle.load(f)

        from train_dive_model import DiveDirectionNet
        self.dive_model = DiveDirectionNet(input_dim=data['feature_dim']).to(self.device)
        self.dive_model.load_state_dict(torch.load(f'{dive_model_dir}/best_model.pth'))
        self.dive_model.eval()

        self.scaler = data['scaler']

        # 关键点映射（YOLO输出顺序 -> 我们的命名）
        self.kp_names = [
            'head', 'shoulde-left', 'shoudle-right',
            'hand-left', 'hand-right',
            'knee-left', 'knee-right',
            'foot-left', 'foot-right'
        ]

        print(f"模型加载完成，设备: {self.device}")

    def extract_features_from_pose(self, keypoints, img_width, img_height):
        """从YOLO姿态结果提取特征"""
        # 构建关键点字典
        kp_dict = {}
        for i, name in enumerate(self.kp_names):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                if conf > 0.3:  # 置信度阈值
                    kp_dict[name] = {'x': x * img_width, 'y': y * img_height}

        # 使用与训练时相同的特征提取逻辑
        features = self._compute_features(kp_dict, img_width, img_height)
        return features, kp_dict

    def _compute_features(self, kps, w, h):
        """计算特征向量（与预处理一致）"""
        features = []

        # 归一化坐标
        coords = []
        for name in self.kp_names:
            if name in kps:
                coords.extend([kps[name]['x'] / w, kps[name]['y'] / h])
            else:
                coords.extend([0, 0])
        features.extend(coords)

        # 辅助函数
        def get(name):
            return (kps[name]['x'] / w, kps[name]['y'] / h) if name in kps else None

        # 头部偏移
        head = get('head')
        sh_l = get('shoulde-left')
        sh_r = get('shoudle-right')
        if head and sh_l and sh_r:
            features.extend([head[0] - (sh_l[0]+sh_r[0])/2, head[1] - (sh_l[1]+sh_r[1])/2])
        else:
            features.extend([0, 0])

        # 手部高度
        hand_l = get('hand-left')
        hand_r = get('hand-right')
        if hand_l and hand_r and sh_l and sh_r:
            features.extend([hand_l[1]-sh_l[1], hand_r[1]-sh_r[1],
                           (hand_l[1]-sh_l[1]) - (hand_r[1]-sh_r[1])])
        else:
            features.extend([0, 0, 0])

        # 膝盖弯曲
        knee_l = get('knee-left')
        knee_r = get('knee-right')
        foot_l = get('foot-left')
        foot_r = get('foot-right')
        if knee_l and knee_r and foot_l and foot_r:
            features.extend([knee_l[1]-foot_l[1], knee_r[1]-foot_r[1],
                          knee_l[0]-foot_l[0], knee_r[0]-foot_r[0]])
        else:
            features.extend([0, 0, 0, 0])

        # 重心和跨度
        valid_x = [c for i, c in enumerate(coords) if i % 2 == 0 and c > 0]
        if valid_x:
            features.append(np.mean(valid_x) - 0.5)
            features.append(max(valid_x) - min(valid_x))
        else:
            features.extend([0, 0])

        return np.array(features, dtype=np.float32)

    def predict(self, image_path, conf_threshold=0.3):
        """
        预测单张图片的扑救方向
        """
        # 1. YOLO姿态估计
        results = self.pose_model(image_path, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return None, "未检测到守门员"

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        predictions = []

        for i, kp_data in enumerate(results.keypoints.data):
            keypoints = kp_data.cpu().numpy()

            # 2. 提取特征
            features, kp_dict = self.extract_features_from_pose(keypoints, w, h)

            # 3. 标准化并预测
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)

            with torch.no_grad():
                output = self.dive_model(features_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = torch.argmax(output).item()

            # 转换回 -1, 0, 1
            direction = pred_class - 1
            direction_names = { -1: 'LEFT', 0: 'CENTER', 1: 'RIGHT' }

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
        """
        可视化预测结果
        """
        predictions, img = self.predict(image_path)

        if predictions is None:
            print("未检测到守门员")
            return

        for pred in predictions:
            kp = pred['keypoints']
            direction = pred['direction_name']
            conf = pred['confidence']

            # 绘制关键点
            for name, coord in kp.items():
                x, y = int(coord['x']), int(coord['y'])
                color = {
                    'head': (0, 0, 255),
                    'shoulde-left': (0, 255, 0), 'shoudle-right': (0, 255, 0),
                    'hand-left': (0, 255, 255), 'hand-right': (0, 255, 255),
                    'knee-left': (255, 0, 0), 'knee-right': (255, 0, 0),
                    'foot-left': (255, 0, 255), 'foot-right': (255, 0, 255)
                }.get(name, (128, 128, 128))
                cv2.circle(img, (x, y), 6, color, -1)

            # 绘制骨骼（简化版）
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
            color_map = {'LEFT': (255, 0, 0), 'CENTER': (0, 255, 0), 'RIGHT': (0, 0, 255)}
            text_color = color_map.get(direction, (255, 255, 255))

            # 文字背景
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
            cv2.putText(img, text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            # 显示概率条
            bar_y = 50
            for j, (name, prob) in enumerate(pred['probabilities'].items()):
                bar_width = int(prob * 200)
                y_pos = bar_y + j * 25
                cv2.rectangle(img, (10, y_pos), (10 + bar_width, y_pos + 20), text_color, -1)
                cv2.putText(img, f"{name}: {prob:.2f}", (220, y_pos + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"结果保存至: {output_path}")

        return img, predictions

def batch_predict(predictor, image_dir, output_dir):
    """批量预测"""
    import os
    from glob import glob

    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(f'{image_dir}/*.jpg') + glob(f'{image_dir}/*.png')

    results = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = f'{output_dir}/pred_{filename}'

        try:
            img, pred = predictor.visualize_prediction(img_path, output_path)
            if pred:
                results.append({
                    'image': filename,
                    'prediction': pred[0]['direction_name'],
                    'confidence': pred[0]['confidence']
                })
        except Exception as e:
            print(f"处理 {filename} 失败: {e}")

    # 保存JSON结果
    with open(f'{output_dir}/predictions.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n批量预测完成，共处理 {len(results)} 张图片")
    return results

if __name__ == '__main__':
    # 模型路径
    pose_model = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\goalkeeper_training\exp_cpu\weights\best.pt'
    dive_model_dir = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction'

    # 初始化预测器
    predictor = GoalkeeperDivePredictor(pose_model, dive_model_dir)

    # 单张测试
    test_image = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames\frame_00023.jpg'
    output = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction\test_result.jpg'

    print("="*50)
    print("单张图片预测")
    print("="*50)

    img, pred = predictor.visualize_prediction(test_image, output)
    if pred:
        print(f"预测结果: {pred[0]['direction_name']}")
        print(f"置信度: {pred[0]['confidence']:.3f}")
        print(f"概率分布: {pred[0]['probabilities']}")

    # 批量预测（可选）
    # batch_predict(
    #     predictor,
    #     r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames',
    #     r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction\batch_results'
    # )

'''
print("\n" + "=" * 80)
print("STEP 3: 端到端推理代码 (predict_dive.py)")
print("=" * 80)
print(inference_code)
'''
