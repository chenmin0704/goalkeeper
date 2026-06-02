# 生成完整的扑救方向预测系统代码

# 1. 数据预处理 - 创建时序特征和方向标签
import json
import numpy as np
import os
import cv2
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_pose_features(keypoints_dict, img_width, img_height):
    """
    从关键点提取姿态特征向量
    """
    features = []

    # 定义关键点顺序（与YOLO一致）
    kp_names = [
        'head', 'shoulde-left', 'shoudle-right',
        'hand-left', 'hand-right',
        'knee-left', 'knee-right',
        'foot-left', 'foot-right'
    ]

    # 归一化坐标
    coords = []
    for name in kp_names:
        if name in keypoints_dict:
            x = keypoints_dict[name]['x'] / img_width
            y = keypoints_dict[name]['y'] / img_height
            coords.extend([x, y])
        else:
            coords.extend([0, 0])  # 缺失点用0填充

    # 基础特征：18维坐标 (9点 x 2)
    features.extend(coords)

    # 几何特征
    def get_coord(name):
        if name in keypoints_dict:
            return (keypoints_dict[name]['x'] / img_width,
                   keypoints_dict[name]['y'] / img_height)
        return None

    # 1. 头部相对肩膀中心的位置（身体倾斜）
    head = get_coord('head')
    sh_l = get_coord('shoulde-left')
    sh_r = get_coord('shoudle-right')
    if head and sh_l and sh_r:
        sh_center_x = (sh_l[0] + sh_r[0]) / 2
        sh_center_y = (sh_l[1] + sh_r[1]) / 2
        features.extend([head[0] - sh_center_x, head[1] - sh_center_y])
    else:
        features.extend([0, 0])

    # 2. 左右手相对肩膀的高度差
    hand_l = get_coord('hand-left')
    hand_r = get_coord('hand-right')
    if hand_l and hand_r and sh_l and sh_r:
        hand_l_rel = hand_l[1] - sh_l[1]  # 左手相对左肩的垂直距离
        hand_r_rel = hand_r[1] - sh_r[1]  # 右手相对右肩的垂直距离
        features.extend([hand_l_rel, hand_r_rel, hand_l_rel - hand_r_rel])
    else:
        features.extend([0, 0, 0])

    # 3. 膝盖相对脚的位置（弯曲程度）
    knee_l = get_coord('knee-left')
    knee_r = get_coord('knee-right')
    foot_l = get_coord('foot-left')
    foot_r = get_coord('foot-right')
    if knee_l and knee_r and foot_l and foot_r:
        knee_l_bend = knee_l[1] - foot_l[1]
        knee_r_bend = knee_r[1] - foot_r[1]
        knee_l_shift = knee_l[0] - foot_l[0]  # 膝盖水平偏移
        knee_r_shift = knee_r[0] - foot_r[0]
        features.extend([knee_l_bend, knee_r_bend, knee_l_shift, knee_r_shift])
    else:
        features.extend([0, 0, 0, 0])

    # 4. 整体重心偏移（所有点的平均x坐标相对于图像中心）
    valid_x = [c for i, c in enumerate(coords) if i % 2 == 0 and c > 0]
    if valid_x:
        center_of_mass = np.mean(valid_x)
        features.append(center_of_mass - 0.5)  # 相对于中心的偏移
    else:
        features.append(0)

    # 5. 身体展开程度（手脚最大跨度）
    if valid_x:
        x_span = max(valid_x) - min(valid_x)
        features.append(x_span)
    else:
        features.append(0)

    return np.array(features, dtype=np.float32)

def label_dive_direction(keypoints_dict, img_width, img_height):
    """
    根据关键点自动标注扑救方向
    返回: -1=左, 0=中/未知, 1=右
    """
    def get(name):
        if name in keypoints_dict:
            return (keypoints_dict[name]['x'] / img_width,
                   keypoints_dict[name]['y'] / img_height)
        return None

    scores = []

    # 指标1: 头部偏移
    head = get('head')
    sh_l = get('shoulde-left')
    sh_r = get('shoudle-right')
    if head and sh_l and sh_r:
        sh_center = (sh_l[0] + sh_r[0]) / 2
        scores.append((head[0] - sh_center) * 3)  # 放大头部权重

    # 指标2: 手部高度不对称（哪只手更低）
    hand_l = get('hand-left')
    hand_r = get('hand-right')
    if hand_l and hand_r:
        # y坐标越大越低，正值表示右手更低（向右扑救）
        scores.append((hand_r[1] - hand_l[1]) * 2)

    # 指标3: 脚部支撑（哪只脚在前）
    foot_l = get('foot-left')
    foot_r = get('foot-right')
    if foot_l and foot_r:
        # 正值表示右脚在前（向右移动）
        scores.append((foot_r[0] - foot_l[0]))

    # 指标4: 整体重心
    all_points = [get(n) for n in ['head', 'shoulde-left', 'shoudle-right',
                                    'hand-left', 'hand-right']]
    valid = [p for p in all_points if p]
    if valid:
        avg_x = np.mean([p[0] for p in valid])
        scores.append((avg_x - 0.5) * 2)

    if not scores:
        return 0

    avg_score = np.mean(scores)

    # 阈值判断
    if avg_score < -0.15:
        return -1  # 向左
    elif avg_score > 0.15:
        return 1   # 向右
    else:
        return 0   # 中立

def prepare_dive_dataset(json_path, output_dir, sequence_length=3):
    """
    准备扑救方向预测数据集
    sequence_length: 使用时序帧数（单帧=1，考虑运动轨迹>1）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总样本: {len(data)}")

    X, y, metadata = [], [], []

    for item in data:
        img_name = item['image']
        width, height = item['width'], item['height']
        kps = item['keypoints']

        # 提取特征
        features = extract_pose_features(kps, width, height)

        # 标注方向
        direction = label_dive_direction(kps, width, height)

        # 只保留有明确方向的样本（可选：保留全部）
        X.append(features)
        y.append(direction)
        metadata.append({
            'image': img_name,
            'direction': direction,
            'direction_name': { -1: 'left', 0: 'center', 1: 'right' }[direction]
        })

    X = np.array(X)
    y = np.array(y)

    print(f"特征维度: {X.shape[1]}")
    print(f"样本分布: 左={sum(y==-1)}, 中={sum(y==0)}, 右={sum(y==1)}")

    # 划分训练集和测试集

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 保存
    dataset = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_dim': X.shape[1],
        'train_metadata': meta_train,
        'test_metadata': meta_test
    }

    with open(os.path.join(output_dir, 'dive_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    print(f"数据集保存至: {output_dir}")
    return dataset

if __name__ == '__main__':
    json_path = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\target\vott\goalkeeper_keypointfinal.json'
    output_dir = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction'

    dataset = prepare_dive_dataset(json_path, output_dir)

'''

print("=" * 80)
print("1. 数据预处理代码 (prepare_dive_data.py)")
print("=" * 80)
print(preprocess_code)
'''