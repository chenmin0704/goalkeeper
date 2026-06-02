# 生成完整的扑救方向预测系统 - 数据预处理
import json
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_pose_features(keypoints_dict, img_width, img_height):
    """从关键点提取姿态特征向量"""
    features = []

    # 定义关键点顺序
    kp_names = [
        'head', 'shoulde-left', 'shoudle-right',
        'hand-left', 'hand-right',
        'knee-left', 'knee-right',
        'foot-left', 'foot-right'
    ]

    # 归一化坐标 (18维)
    coords = []
    for name in kp_names:
        if name in keypoints_dict:
            x = keypoints_dict[name]['x'] / img_width
            y = keypoints_dict[name]['y'] / img_height
            coords.extend([x, y])
        else:
            coords.extend([0, 0])
    features.extend(coords)

    # 辅助函数
    def get_coord(name):
        if name in keypoints_dict:
            return (keypoints_dict[name]['x'] / img_width,
                   keypoints_dict[name]['y'] / img_height)
        return None

    # 1. 头部相对肩膀中心 (2维)
    head = get_coord('head')
    sh_l = get_coord('shoulde-left')
    sh_r = get_coord('shoudle-right')
    if head and sh_l and sh_r:
        sh_center_x = (sh_l[0] + sh_r[0]) / 2
        sh_center_y = (sh_l[1] + sh_r[1]) / 2
        features.extend([head[0] - sh_center_x, head[1] - sh_center_y])
    else:
        features.extend([0, 0])

    # 2. 手部相对高度 (3维)
    hand_l = get_coord('hand-left')
    hand_r = get_coord('hand-right')
    if hand_l and hand_r and sh_l and sh_r:
        hand_l_rel = hand_l[1] - sh_l[1]
        hand_r_rel = hand_r[1] - sh_r[1]
        features.extend([hand_l_rel, hand_r_rel, hand_l_rel - hand_r_rel])
    else:
        features.extend([0, 0, 0])

    # 3. 膝盖弯曲 (4维)
    knee_l = get_coord('knee-left')
    knee_r = get_coord('knee-right')
    foot_l = get_coord('foot-left')
    foot_r = get_coord('foot-right')
    if knee_l and knee_r and foot_l and foot_r:
        features.extend([
            knee_l[1] - foot_l[1],  # 左膝弯曲
            knee_r[1] - foot_r[1],  # 右膝弯曲
            knee_l[0] - foot_l[0],  # 左膝水平偏移
            knee_r[0] - foot_r[0]   # 右膝水平偏移
        ])
    else:
        features.extend([0, 0, 0, 0])

    # 4. 重心偏移 (1维)
    valid_x = [c for i, c in enumerate(coords) if i % 2 == 0 and c > 0]
    if valid_x:
        features.append(np.mean(valid_x) - 0.5)
    else:
        features.append(0)

    # 5. 身体跨度 (1维)
    if valid_x:
        features.append(max(valid_x) - min(valid_x))
    else:
        features.append(0)

    return np.array(features, dtype=np.float32)

def label_dive_direction(keypoints_dict, img_width, img_height):
    """自动标注扑救方向: -1=左, 0=中, 1=右"""
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
        scores.append((head[0] - sh_center) * 3)

    # 指标2: 手部高度差
    hand_l = get('hand-left')
    hand_r = get('hand-right')
    if hand_l and hand_r:
        scores.append((hand_r[1] - hand_l[1]) * 2)

    # 指标3: 脚部位置
    foot_l = get('foot-left')
    foot_r = get('foot-right')
    if foot_l and foot_r:
        scores.append((foot_r[0] - foot_l[0]))

    # 指标4: 整体重心
    all_pts = [get(n) for n in ['head', 'shoulde-left', 'shoudle-right',
                                 'hand-left', 'hand-right']]
    valid = [p for p in all_pts if p]
    if valid:
        scores.append((np.mean([p[0] for p in valid]) - 0.5) * 2)

    if not scores:
        return 0

    avg = np.mean(scores)
    if avg < -0.15: return -1
    elif avg > 0.15: return 1
    else: return 0

def prepare_dataset(json_path, output_dir):
    """准备数据集"""
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总样本: {len(data)}")

    X, y, meta = [], [], []

    for item in data:
        kps = item['keypoints']
        w, h = item['width'], item['height']

        features = extract_pose_features(kps, w, h)
        direction = label_dive_direction(kps, w, h)

        X.append(features)
        y.append(direction)
        meta.append({
            'image': item['image'],
            'direction': direction,
            'name': {-1: 'left', 0: 'center', 1: 'right'}[direction]
        })

    X, y = np.array(X), np.array(y)

    # 统计
    print(f"\n分布: 左={sum(y==-1)}, 中={sum(y==0)}, 右={sum(y==1)}")

    # 划分数据集
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 保存
    with open(os.path.join(output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'train_meta': m_train, 'test_meta': m_test,
            'scaler': scaler, 'feature_dim': X.shape[1]
        }, f)

    print(f"保存至: {output_dir}")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    json_path = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\target\vott\goalkeeper_keypointfinal.json'
    output_dir = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction'
    prepare_dataset(json_path, output_dir)

