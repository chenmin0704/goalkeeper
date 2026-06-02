"""
将 goalkeeper_keypoints.json 转换为 YOLOv8-Pose 训练格式
"""

import json
import os
import shutil
from pathlib import Path
import random

# ==================== 配置 ====================
JSON_FILE = r"C:\Users\chen3\Desktop\ok\新建文件夹\goalkeeper_keypoints.json"
IMAGE_DIR = r"C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames"  # 图像所在目录
OUTPUT_DIR = r"C:\Users\chen3\Desktop\ok\新建文件夹\yolo_pose_dataset"
TRAIN_RATIO = 0.8
# =============================================

# 关键点顺序（YOLOv8-Pose要求固定顺序）
KEYPOINT_ORDER = [
    'head',
    'shoulder-left',
    'shoulder-right',
    'hand-left',
    'hand-right',
    'knee-left',
    'knee-right',
    'foot-left',
    'foot-right'
]


def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_bbox(keypoints, img_w, img_h):
    """
    基于所有可见关键点计算边界框
    返回: (xc, yc, w, h) 归一化值
    """
    if not keypoints:
        return 0.5, 0.5, 1.0, 1.0

    xs = [kp['x'] for kp in keypoints.values()]
    ys = [kp['y'] for kp in keypoints.values()]

    # 添加边距
    margin = 20
    x_min = max(0, min(xs) - margin)
    y_min = max(0, min(ys) - margin)
    x_max = min(img_w, max(xs) + margin)
    y_max = min(img_h, max(ys) + margin)

    w = x_max - x_min
    h = y_max - y_min
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2

    # 归一化
    return xc / img_w, yc / img_h, w / img_w, h / img_h


def convert_sample(item):
    """
    将单个JSON样本转换为YOLOv8-Pose格式字符串
    """
    img_w = item['width']
    img_h = item['height']
    kps = item['keypoints']

    # 计算bbox（基于所有关键点）
    xc, yc, bw, bh = compute_bbox(kps, img_w, img_h)

    # class = 0（只有一个类别：goalkeeper）
    values = [0, xc, yc, bw, bh]

    # 按固定顺序输出9个关键点
    for kp_name in KEYPOINT_ORDER:
        if kp_name in kps:
            kp = kps[kp_name]
            values.extend([kp['x'] / img_w, kp['y'] / img_h, 2])  # 2 = 可见
        else:
            values.extend([0, 0, 0])  # 0,0,0 = 缺失

    return ' '.join(f'{v:.6f}' for v in values)


def split_data(data, ratio, seed=42):
    """划分训练集和验证集"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * ratio)
    return shuffled[:n_train], shuffled[n_train:]


def setup_dirs(output_dir):
    """创建目录结构"""
    dirs = {
        'images_train': os.path.join(output_dir, 'images', 'train'),
        'images_val': os.path.join(output_dir, 'images', 'val'),
        'labels_train': os.path.join(output_dir, 'labels', 'train'),
        'labels_val': os.path.join(output_dir, 'labels', 'val'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def process_split(data_split, dirs, split_name, image_dir):
    """处理一个数据划分"""
    img_out = dirs[f'images_{split_name}']
    lbl_out = dirs[f'labels_{split_name}']

    success = 0
    missing_img = 0

    for item in data_split:
        img_name = item['image']
        base_name = Path(img_name).stem

        # 查找图像文件
        src_img = os.path.join(image_dir, img_name)
        if not os.path.exists(src_img):
            # 尝试其他扩展名
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                alt = os.path.join(image_dir, base_name + ext)
                if os.path.exists(alt):
                    src_img = alt
                    img_name = base_name + ext
                    found = True
                    break
            if not found:
                missing_img += 1
                continue

        # 复制图像
        dst_img = os.path.join(img_out, img_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)

        # 生成YOLO标签文件
        yolo_line = convert_sample(item)
        txt_path = os.path.join(lbl_out, base_name + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(yolo_line + '\n')

        success += 1

    return success, missing_img


def create_data_yaml(output_dir):
    """生成YOLOv8-Pose的data.yaml"""
    yaml_content = f"""# YOLOv8-Pose 守门员关键点检测
path: {output_dir.replace(chr(92), '/')}

train: images/train
val: images/val

kpt_shape: [9, 3]  # 9个关键点，每个 (x, y, visibility)

# 左右翻转对称索引
# 0-head, 1-shoulder-left, 2-shoulder-right
# 3-hand-left, 4-hand-right
# 5-knee-left, 6-knee-right
# 7-foot-left, 8-foot-right
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7]

names:
  0: goalkeeper
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    return yaml_path


def main():
    print("=" * 60)
    print("JSON -> YOLOv8-Pose 格式转换")
    print("=" * 60)

    # 加载数据
    print(f"\n加载: {JSON_FILE}")
    data = load_json(JSON_FILE)
    print(f"总样本: {len(data)}")

    # 划分数据集
    train_data, val_data = split_data(data, TRAIN_RATIO)
    print(f"训练集: {len(train_data)}  |  验证集: {len(val_data)}")

    # 创建目录
    dirs = setup_dirs(OUTPUT_DIR)

    # 处理训练集
    print("\n处理训练集...")
    n_train, miss_train = process_split(train_data, dirs, 'train', IMAGE_DIR)
    print(f"  成功: {n_train}, 图像缺失: {miss_train}")

    # 处理验证集
    print("处理验证集...")
    n_val, miss_val = process_split(val_data, dirs, 'val', IMAGE_DIR)
    print(f"  成功: {n_val}, 图像缺失: {miss_val}")

    # 生成data.yaml
    yaml_path = create_data_yaml(OUTPUT_DIR)
    print(f"\ndata.yaml: {yaml_path}")

    # 统计
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"  images/train: {n_train}张")
    print(f"  images/val: {n_val}张")
    print(f"  labels/train: {n_train}个")
    print(f"  labels/val: {n_val}个")

    # 验证一个样本
    if train_data:
        print("\n样本验证:")
        sample = convert_sample(train_data[0])
        parts = sample.split()
        print(f"  数值个数: {len(parts)} (1+4+9*3=32)")
        print(f"  class={parts[0]}, bbox=({parts[1]},{parts[2]},{parts[3]},{parts[4]})")
        print(f"  前3个关键点:")
        for i in range(3):
            idx = 5 + i * 3
            print(f"    {KEYPOINT_ORDER[i]}: x={parts[idx]}, y={parts[idx + 1]}, v={parts[idx + 2]}")


if __name__ == '__main__':
    main()