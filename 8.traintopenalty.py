# 生成完整的扑救方向预测代码
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class GoalkeeperDivePredictor:
    """
    守门员扑救方向预测器
    基于关键点预测：左扑(left)、右扑(right)、中立(center)
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.feature_names = [
            'body_tilt', 'body_tilt_abs',
            'hand_diff_x', 'hand_diff_y', 'hand_spread_left', 'hand_spread_right', 'hand_spread_both',
            'foot_diff_x', 'foot_diff_y', 'lead_foot_left', 'lead_foot_right',
            'head_offset', 'head_offset_abs',
            'body_height', 'body_width',
            'left_hand_x', 'right_hand_x', 'left_hand_y', 'right_hand_y',
            'left_foot_x', 'right_foot_x', 'left_foot_y', 'right_foot_y',
            'shoulder_width', 'hip_width'
        ]

    def extract_features(self, keypoints, img_w, img_h):
        """
        从关键点提取特征向量
        """
        def get_kp(name, default=None):
            if name in keypoints:
                return np.array([
                    keypoints[name]['x'] / img_w,  # 归一化x
                    keypoints[name]['y'] / img_h   # 归一化y
                ])
            return default

        # 获取关键点
        head = get_kp('head', np.array([0.5, 0.3]))
        sh_l = get_kp('shoulde-left')
        sh_r = get_kp('shoudle-right')
        hand_l = get_kp('hand-left')
        hand_r = get_kp('hand-right')
        knee_l = get_kp('knee-left')
        knee_r = get_kp('knee-right')
        foot_l = get_kp('foot-left')
        foot_r = get_kp('foot-right')

        features = {}

        # 1. 身体中心点
        if sh_l is not None and sh_r is not None:
            shoulder_center = (sh_l + sh_r) / 2
            features['shoulder_width'] = abs(sh_r[0] - sh_l[0])
        else:
            shoulder_center = head
            features['shoulder_width'] = 0.1

        if knee_l is not None and knee_r is not None:
            hip_center = (knee_l + knee_r) / 2
            features['hip_width'] = abs(knee_r[0] - knee_l[0])
        else:
            hip_center = shoulder_center
            features['hip_width'] = 0.1

        # 2. 身体倾斜（关键特征）
        features['body_tilt'] = shoulder_center[0] - hip_center[0]
        features['body_tilt_abs'] = abs(features['body_tilt'])

        # 3. 双手特征
        if hand_l is not None and hand_r is not None:
            features['hand_diff_x'] = hand_r[0] - hand_l[0]
            features['hand_diff_y'] = hand_r[1] - hand_l[1]
            features['left_hand_x'] = hand_l[0]
            features['right_hand_x'] = hand_r[0]
            features['left_hand_y'] = hand_l[1]
            features['right_hand_y'] = hand_r[1]

            # 哪只手更外侧
            features['hand_spread_left'] = 1 if hand_l[0] < shoulder_center[0] - 0.15 else 0
            features['hand_spread_right'] = 1 if hand_r[0] > shoulder_center[0] + 0.15 else 0
            features['hand_spread_both'] = 1 if (features['hand_spread_left'] and features['hand_spread_right']) else 0
        else:
            features['hand_diff_x'] = 0
            features['hand_diff_y'] = 0
            features['left_hand_x'] = shoulder_center[0] - 0.1
            features['right_hand_x'] = shoulder_center[0] + 0.1
            features['left_hand_y'] = shoulder_center[1]
            features['right_hand_y'] = shoulder_center[1]
            features['hand_spread_left'] = 0
            features['hand_spread_right'] = 0
            features['hand_spread_both'] = 0

        # 4. 双脚特征
        if foot_l is not None and foot_r is not None:
            features['foot_diff_x'] = foot_r[0] - foot_l[0]
            features['foot_diff_y'] = foot_r[1] - foot_l[1]
            features['left_foot_x'] = foot_l[0]
            features['right_foot_x'] = foot_r[0]
            features['left_foot_y'] = foot_l[1]
            features['right_foot_y'] = foot_r[1]

            # 哪只脚在前（y值小表示靠上/前）
            features['lead_foot_left'] = 1 if foot_l[1] < foot_r[1] - 0.03 else 0
            features['lead_foot_right'] = 1 if foot_r[1] < foot_l[1] - 0.03 else 0
        else:
            features['foot_diff_x'] = 0
            features['foot_diff_y'] = 0
            features['left_foot_x'] = hip_center[0] - 0.05
            features['right_foot_x'] = hip_center[0] + 0.05
            features['left_foot_y'] = hip_center[1] + 0.2
            features['right_foot_y'] = hip_center[1] + 0.2
            features['lead_foot_left'] = 0
            features['lead_foot_right'] = 0

        # 5. 头部偏移
        features['head_offset'] = head[0] - shoulder_center[0]
        features['head_offset_abs'] = abs(features['head_offset'])

        # 6. 身体尺寸
        features['body_height'] = hip_center[1] - head[1]
        features['body_width'] = max(features['shoulder_width'], features['hip_width'])

        # 构建特征向量
        feature_vector = [features[name] for name in self.feature_names]
        return np.array(feature_vector), features

    def rule_based_predict(self, features_dict):
        """
        基于规则的预测（用于对比或初始标注）
        """
        score = 0
        reasons = []

        # 身体倾斜（权重最高）
        tilt = features_dict['body_tilt']
        if tilt < -0.03:
            score -= 3
            reasons.append(f"身体明显左倾({tilt:.3f})")
        elif tilt < -0.01:
            score -= 1
            reasons.append(f"身体轻微左倾({tilt:.3f})")
        elif tilt > 0.03:
            score += 3
            reasons.append(f"身体明显右倾({tilt:.3f})")
        elif tilt > 0.01:
            score += 1
            reasons.append(f"身体轻微右倾({tilt:.3f})")

        # 双手位置
        hand_diff = features_dict['hand_diff_x']
        if hand_diff < -0.05:
            score -= 2
            reasons.append(f"左手更外侧({hand_diff:.3f})")
        elif hand_diff > 0.05:
            score += 2
            reasons.append(f"右手更外侧({hand_diff:.3f})")

        # 单手伸展
        if features_dict['hand_spread_left'] and not features_dict['hand_spread_right']:
            score -= 2
            reasons.append("左手单独伸展")
        elif features_dict['hand_spread_right'] and not features_dict['hand_spread_left']:
            score += 2
            reasons.append("右手单独伸展")

        # 头部偏移
        head_off = features_dict['head_offset']
        if head_off < -0.03:
            score -= 1
            reasons.append("头部左偏")
        elif head_off > 0.03:
            score += 1
            reasons.append("头部右偏")

        # 判断方向
        if score <= -3:
            return 'left', score, reasons
        elif score >= 3:
            return 'right', score, reasons
        else:
            return 'center', score, reasons

    def prepare_training_data(self, json_path, manual_labels=None):
        """
        准备训练数据
        manual_labels: 可选的手动标注字典 {image_name: direction}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        X = []
        y = []
        metadata = []

        for item in data:
            kp = item['keypoints']
            if len(kp) < 5:  # 跳过关键点太少的
                continue

            w, h = item['width'], item['height']
            feat_vec, feat_dict = self.extract_features(kp, w, h)

            # 获取标签
            if manual_labels and item['image'] in manual_labels:
                label = manual_labels[item['image']]
            else:
                # 使用规则自动标注
                label, _, _ = self.rule_based_predict(feat_dict)

            X.append(feat_vec)
            y.append(label)
            metadata.append({
                'image': item['image'],
                'features': feat_dict,
                'label': label
            })

        return np.array(X), np.array(y), metadata

    def train(self, X, y):
        """
        训练分类模型
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练随机森林
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # 评估
        y_pred = self.model.predict(X_test)
        print("="*50)
        print("模型评估:")
        print(classification_report(y_test, y_pred))
        print("\\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred, labels=['left', 'center', 'right']))

        # 特征重要性
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\\nTop 10 重要特征:")
        for i in range(min(10, len(importances))):
            print(f"  {i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.3f}")

        return self.model

    def predict(self, keypoints, img_w, img_h):
        """
        预测单个样本
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用train()")

        feat_vec, feat_dict = self.extract_features(keypoints, img_w, img_h)
        feat_vec = feat_vec.reshape(1, -1)

        prediction = self.model.predict(feat_vec)[0]
        probabilities = self.model.predict_proba(feat_vec)[0]

        # 获取类别顺序
        classes = self.model.classes_
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}

        return prediction, prob_dict, feat_dict

    def save(self, path):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
        print(f"模型保存至: {path}")

    def load(self, path):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        print(f"模型加载自: {path}")


def visualize_prediction(image_path, keypoints, prediction, probs, output_path=None):
    """
    可视化预测结果
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # 绘制关键点（简化版）
    kp_colors = {
        'head': (0, 0, 255),
        'shoulde-left': (0, 255, 0),
        'shoudle-right': (0, 255, 0),
        'hand-left': (0, 255, 255),
        'hand-right': (0, 255, 255),
        'knee-left': (255, 0, 0),
        'knee-right': (255, 0, 0),
        'foot-left': (255, 0, 255),
        'foot-right': (255, 0, 255),
    }

    for kp_name, kp_data in keypoints.items():
        if kp_name == 'gialkeeper':
            continue
        x = int(kp_data['x'])
        y = int(kp_data['y'])
        color = kp_colors.get(kp_name, (128, 128, 128))
        cv2.circle(img, (x, y), 8, color, -1)

    # 显示预测结果
    result_text = f"Dive: {prediction.upper()}"
    prob_text = f"L:{probs.get('left', 0):.2f} C:{probs.get('center', 0):.2f} R:{probs.get('right', 0):.2f}"

    # 背景框
    cv2.rectangle(img, (10, 10), (400, 80), (0, 0, 0), -1)

    # 文字颜色根据预测结果
    if prediction == 'left':
        color = (255, 0, 0)  # 蓝色（左）
    elif prediction == 'right':
        color = (0, 0, 255)  # 红色（右）
    else:
        color = (0, 255, 0)  # 绿色（中）

    cv2.putText(img, result_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(img, prob_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if output_path:
        cv2.imwrite(output_path, img)

    return img


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 路径配置
    json_path = r'C:\\Users\\chen3\\Desktop\\ok\\datasets\\goalkeeper\\target\\vott\\goalkeeper_keypointfinal.json'
    image_dir = r'C:\\Users\\chen3\\Desktop\\ok\\datasets\\goalkeeper\\all_frames'
    model_path = r'C:\\Users\\chen3\\Desktop\\ok\\datasets\\goalkeeper\\dive_predictor.pkl'

    # 初始化
    predictor = GoalkeeperDivePredictor()

    # 准备数据（可以传入手动标注来改进准确性）
    # manual_labels = {'frame_00023.jpg': 'right', 'frame_00031.jpg': 'left', ...}
    X, y, metadata = predictor.prepare_training_data(json_path, manual_labels=None)

    print(f"训练样本数: {len(X)}")
    print(f"类别分布: left={sum(y=='left')}, center={sum(y=='center')}, right={sum(y=='right')}")

    # 训练
    predictor.train(X, y)

    # 保存模型
    predictor.save(model_path)

    # 测试预测
    print("\\n" + "="*50)
    print("测试预测:")

    # 读取JSON进行测试
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    for item in test_data[:5]:
        if len(item['keypoints']) < 5:
            continue

        pred, probs, feats = predictor.predict(item['keypoints'], item['width'], item['height'])
        print(f"\\n{item['image']}:")
        print(f"  预测: {pred}")
        print(f"  概率: {probs}")
        print(f"  身体倾斜: {feats['body_tilt']:.3f}")
        print(f"  手差异: {feats['hand_diff_x']:.3f}")

        # 可视化（可选）
        # img_path = os.path.join(image_dir, item['image'])
        # out_path = os.path.join(image_dir, '..', 'predictions', item['image'])
        # visualize_prediction(img_path, item['keypoints'], pred, probs, out_path)
'''

print(script_content)
print("\n" + "=" * 80)
print("请将上述代码保存为 dive_predictor.py 并运行")
print("=" * 80)
'''