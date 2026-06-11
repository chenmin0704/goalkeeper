"""
VoTT 标注转换为 COCO 和 YOLO (YOLOv8-Pose) 格式
修复肩膀标签匹配问题，支持关键点检测训练

作者: Assistant
日期: 2025-04-21
"""

import json
import os
import glob
import shutil
import random
from pathlib import Path
from collections import defaultdict


# ============================================================
# 配置区 - 根据你的项目修改这些参数
# ============================================================

# VoTT 项目文件路径
VOTT_FILE = r"C:\Users\chen3\Desktop\ok\新建文件夹\new.vott"

# 输出目录
OUTPUT_DIR = r"C:\Users\chen3\Desktop\ok\新建文件夹\dataset_output"

# 数据集划分比例 (train / val / test)
# 例如: [0.8, 0.2] 表示只分 train 和 val
# 例如: [0.7, 0.2, 0.1] 表示 train / val / test
SPLIT_RATIOS = [0.8, 0.2]

# 随机种子，保证划分结果可复现
RANDOM_SEED = 42

# 图像文件扩展名
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# 关键点定义（使用标准英文拼写）
# 注意：这是修复后的标准命名，会自动映射你VoTT中的各种拼写变体
KEYPOINT_NAMES = [
    'head',
    'shoulder-left',    # 修复: 原 VoTT 中可能是 'shoulde-left ', 'shoulde-left'
    'shoulder-right',   # 修复: 原 VoTT 中可能是 'shoulde-lright', 'shoudle-right'
    'hand-left',
    'hand-right',
    'knee-left',
    'knee-right',
    'foot-left',
    'foot-right'
]

# 骨骼连接定义（用于可视化）
SKELETON = [
    [0, 1], [0, 2],       # head -> shoulders
    [1, 3], [2, 4],       # shoulders -> hands
    [1, 5], [2, 6],       # shoulders -> knees
    [5, 7], [6, 8]        # knees -> feet
]

# 类别名称
CLASS_NAME = 'goalkeeper'
NUM_CLASSES = 1


# ============================================================
# 标签名称映射表 - 处理 VoTT 中的拼写不一致
# ============================================================

# 这是核心修复：将你在 VoTT 中实际使用的各种标签名变体映射到标准名称
# 键: VoTT 中的原始标签名（小写、去空格后的形式）
# 值: 映射后的标准关键点名称
LABEL_MAPPING = {
    # 头部
    'head': 'head',

    # 左肩 - 处理各种拼写变体
    'shoulde-left': 'shoulder-left',      # 你在 VoTT 中的写法（正确映射）
    'shoulde-left ': 'shoulder-left',     # 带尾部空格的变体
    'shoulder-left': 'shoulder-left',     # 标准写法
    'shoudle-left': 'shoulder-left',      # 拼写错误的变体

    # 右肩 - 处理各种拼写变体
    'shoudle-right': 'shoulder-right',    # 代码中期望的拼写
    'shoulde-right': 'shoulder-right',    # 另一种常见拼写
    'shoulde-lright': 'shoulder-right',   # 你在 VoTT 中的实际写法！
    'shoulder-right': 'shoulder-right',   # 标准写法
    'shouder-right': 'shoulder-right',    # 少了一个 l

    # 手部
    'hand-left': 'hand-left',
    'hand-right': 'hand-right',

    # 膝盖
    'knee-left': 'knee-left',
    'knee-right': 'knee-right',

    # 脚部
    'foot-left': 'foot-left',
    'foot-right': 'foot-right',

    # 以下标签将被忽略（不是关键点）
    'goalkeeper': None,
}


def normalize_label(tag):
    """
    标准化标签名：去除首尾空格，转为小写
    """
    if not tag:
        return ''
    return tag.strip().lower()


def map_label(tag):
    """
    将 VoTT 中的标签名映射到标准关键点名称
    返回标准名称，如果无法映射或应忽略则返回 None
    """
    normalized = normalize_label(tag)

    # 直接查映射表
    if normalized in LABEL_MAPPING:
        return LABEL_MAPPING[normalized]

    # 尝试模糊匹配：去除所有空格和连字符再比较
    tag_compact = normalized.replace(' ', '').replace('-', '')
    for key, value in LABEL_MAPPING.items():
        key_compact = key.replace(' ', '').replace('-', '')
        if tag_compact == key_compact:
            return value

    # 无法映射
    return None


# ============================================================
# VoTT 解析函数
# ============================================================

def parse_vott_project(vott_file_path):
    """
    解析 .vott 项目文件，获取标注文件夹路径
    """
    with open(vott_file_path, 'r', encoding='utf-8') as f:
        project = json.load(f)

    print(f"项目名: {project.get('name', 'Unknown')}")

    # 获取 target 连接路径（标注文件所在目录）
    target_connection = project.get('targetConnection', {})
    provider_options = target_connection.get('providerOptions', {})
    target_folder = provider_options.get('folderPath', '')

    # 获取 source 连接路径（源图像目录）
    source_connection = project.get('sourceConnection', {})
    source_options = source_connection.get('providerOptions', {})
    source_folder = source_options.get('folderPath', '')

    print(f"源图像文件夹: {source_folder}")
    print(f"标注文件夹: {target_folder}")
    print(f"标签列表: {[tag['name'] for tag in project.get('tags', [])]}")

    # 如果 target_folder 无效，使用 .vott 所在目录
    if not target_folder or not os.path.exists(target_folder):
        vott_dir = os.path.dirname(vott_file_path)
        print(f"使用 .vott 所在目录作为标注文件夹: {vott_dir}")
        target_folder = vott_dir

    return target_folder, source_folder, project.get('tags', [])


def find_asset_files(target_folder):
    """
    查找所有 VoTT 标注文件 (*-asset.json)
    """
    # 首先查找 *-asset.json 文件
    asset_files = glob.glob(os.path.join(target_folder, '*-asset.json'))

    if not asset_files:
        # 回退：查找所有 json 文件（排除 .vott）
        all_json = glob.glob(os.path.join(target_folder, '*.json'))
        asset_files = [f for f in all_json if not f.endswith('.vott')]

    if not asset_files:
        raise FileNotFoundError(f"在 {target_folder} 中找不到任何标注文件")

    print(f"找到 {len(asset_files)} 个标注文件")
    return sorted(asset_files)


def parse_asset_file(asset_file):
    """
    解析单个 asset 文件，提取图像信息和关键点
    返回: (image_name, width, height, keypoints_dict, warnings)
    """
    with open(asset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    asset = data.get('asset', {})
    image_name = asset.get('name', '')
    width = asset.get('size', {}).get('width', 0)
    height = asset.get('size', {}).get('height', 0)

    # 收集关键点
    keypoints_dict = {}
    warnings = []

    for region in data.get('regions', []):
        tags_list = region.get('tags', [])
        if not tags_list:
            continue

        # 取第一个标签
        raw_tag = tags_list[0]
        bbox = region.get('boundingBox', {})

        if not bbox:
            continue

        # 映射标签名
        mapped_tag = map_label(raw_tag)

        if mapped_tag is None:
            # 这个标签被忽略（如 'goalkeeper'）
            continue

        # 计算中心点（关键点位置）
        cx = bbox.get('left', 0) + bbox.get('width', 0) / 2
        cy = bbox.get('top', 0) + bbox.get('height', 0) / 2

        # 检查重复
        if mapped_tag in keypoints_dict:
            warnings.append(f"重复关键点 '{mapped_tag}'，保留第一个")
            continue

        keypoints_dict[mapped_tag] = {
            'x': round(cx, 2),
            'y': round(cy, 2),
            'visibility': 2  # 2 = 可见且已标注
        }

    return image_name, width, height, keypoints_dict, warnings


# ============================================================
# COCO 格式生成
# ============================================================

def build_coco_data(samples):
    """
    构建 COCO 格式数据结构
    """
    coco_data = {
        'info': {
            'description': 'Goalkeeper Keypoint Dataset',
            'version': '1.1',
            'year': 2025,
            'contributor': 'VoTT Converter'
        },
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1,
            'name': CLASS_NAME,
            'keypoints': KEYPOINT_NAMES,
            'skeleton': SKELETON
        }]
    }

    annotation_id = 1

    for idx, sample in enumerate(samples):
        image_name = sample['image_name']
        width = sample['width']
        height = sample['height']
        keypoints_dict = sample['keypoints']

        if not keypoints_dict:
            continue

        image_id = idx + 1

        # 添加图像信息
        coco_data['images'].append({
            'id': image_id,
            'file_name': image_name,
            'width': width,
            'height': height
        })

        # 构建关键点数组 [x, y, visibility, x, y, visibility, ...]
        keypoints_list = []
        num_keypoints = 0

        for kp_name in KEYPOINT_NAMES:
            if kp_name in keypoints_dict:
                kp = keypoints_dict[kp_name]
                keypoints_list.extend([kp['x'], kp['y'], kp['visibility']])
                num_keypoints += 1
            else:
                # 缺失的关键点
                keypoints_list.extend([0, 0, 0])

        # 计算边界框
        visible_kps = [keypoints_dict[k] for k in keypoints_dict]
        if visible_kps:
            xs = [kp['x'] for kp in visible_kps]
            ys = [kp['y'] for kp in visible_kps]

            # 添加小边距
            margin = 10
            x_min = max(0, min(xs) - margin)
            y_min = max(0, min(ys) - margin)
            x_max = min(width, max(xs) + margin)
            y_max = min(height, max(ys) + margin)

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = bbox[2] * bbox[3]
        else:
            bbox = [0, 0, width, height]
            area = width * height

        # 添加标注
        coco_data['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,
            'bbox': [round(x, 2) for x in bbox],
            'area': round(area, 2),
            'keypoints': keypoints_list,
            'num_keypoints': num_keypoints,
            'iscrowd': 0,
            'segmentation': []  # COCO标准要求，留空
        })

        annotation_id += 1

    return coco_data


def save_coco_json(coco_data, output_path):
    """
    保存 COCO 格式 JSON 文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    print(f"COCO 格式已保存: {output_path}")
    return output_path


# ============================================================
# YOLO (YOLOv8-Pose) 格式生成
# ============================================================

def build_yolo_data(sample):
    """
    将单个样本转换为 YOLOv8-Pose 格式的一行文本
    格式: <class> <x_center> <y_center> <width> <height> <px1> <py1> <pv1> <px2> <py2> <pv2> ...
    所有坐标归一化到 0-1
    """
    width = sample['width']
    height = sample['height']
    keypoints_dict = sample['keypoints']

    if width == 0 or height == 0:
        return None

    # 计算边界框（基于所有可见关键点）
    if keypoints_dict:
        xs = [kp['x'] for kp in keypoints_dict.values()]
        ys = [kp['y'] for kp in keypoints_dict.values()]

        # 添加边距
        margin = 10
        x_min = max(0, min(xs) - margin)
        y_min = max(0, min(ys) - margin)
        x_max = min(width, max(xs) + margin)
        y_max = min(height, max(ys) + margin)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        bbox_xc = (x_min + x_max) / 2
        bbox_yc = (y_min + y_max) / 2

        # 归一化
        bbox_xc_norm = bbox_xc / width
        bbox_yc_norm = bbox_yc / height
        bbox_w_norm = bbox_w / width
        bbox_h_norm = bbox_h / height
    else:
        bbox_xc_norm = 0.5
        bbox_yc_norm = 0.5
        bbox_w_norm = 1.0
        bbox_h_norm = 1.0

    # 构建关键点部分
    yolo_values = [0,  # class id (只有一个类别)
                   bbox_xc_norm, bbox_yc_norm, bbox_w_norm, bbox_h_norm]

    for kp_name in KEYPOINT_NAMES:
        if kp_name in keypoints_dict:
            kp = keypoints_dict[kp_name]
            px_norm = kp['x'] / width
            py_norm = kp['y'] / height
            pv = kp['visibility']  # 2 = 可见
            yolo_values.extend([px_norm, py_norm, pv])
        else:
            # 缺失的关键点
            yolo_values.extend([0, 0, 0])

    # 格式化为字符串（保留6位小数）
    line = ' '.join(f'{v:.6f}' for v in yolo_values)
    return line


def save_yolo_labels(samples, output_dir):
    """
    保存 YOLO 格式标签文件
    每张图像对应一个 .txt 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    for sample in samples:
        line = build_yolo_data(sample)
        if line is None:
            continue

        # 用图像名（去掉扩展名）作为txt文件名
        image_name = sample['image_name']
        base_name = os.path.splitext(image_name)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(line + '\n')
        saved_count += 1

    print(f"YOLO 标签已保存: {saved_count} 个文件到 {output_dir}")
    return saved_count


def create_yolo_yaml(output_dir, class_name, num_keypoints):
    """
    创建 YOLO 训练所需的 data.yaml 文件
    """
    yaml_content = f"""# YOLOv8-Pose 数据集配置
# 由 VoTT 转换脚本自动生成

# 数据集路径
path: {output_dir.replace('\\', '/')}  # 数据集根目录

# 训练/验证/测试 图像路径
train: images/train
val: images/val
# test: images/test  # 如需测试集请取消注释

# 关键点配置
kpt_shape: [{num_keypoints}, 3]  # 9个关键点，每个 (x, y, visibility)

# 是否翻转关键点了（左右对称翻转）
# 格式: [源关键点索引, 目标关键点索引]
flip_idx: [0, 2, 1, 5, 4, 3, 7, 6, 8]
# 说明:
#   0-head 保持 (0->0)
#   1-shoulder-left <-> 2-shoulder-right (1<->2)
#   3-hand-left <-> 4-hand-right (3<->4)
#   5-knee-left <-> 6-knee-right (5<->6)
#   7-foot-left <-> 8-foot-right (7<->8)

# 类别
names:
  0: {class_name}

# 可选：骨骼连接（用于可视化，与COCO skeleton一致）
skeleton: {SKELETON}
"""

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"YOLO data.yaml 已保存: {yaml_path}")
    return yaml_path


# ============================================================
# 数据集划分
# ============================================================

def split_dataset(samples, ratios, seed=42):
    """
    将数据集划分为 train / val / (test)
    ratios: 划分比例列表，如 [0.8, 0.2] 或 [0.7, 0.2, 0.1]
    """
    random.seed(seed)

    # 复制一份避免修改原列表
    data = samples.copy()
    random.shuffle(data)

    n = len(data)
    if len(ratios) == 2:
        # train / val
        n_train = int(n * ratios[0])
        train_set = data[:n_train]
        val_set = data[n_train:]
        test_set = []
    elif len(ratios) == 3:
        # train / val / test
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_set = data[:n_train]
        val_set = data[n_train:n_train + n_val]
        test_set = data[n_train + n_val:]
    else:
        raise ValueError("ratios 必须是长度为 2 或 3 的列表")

    return {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }


def setup_yolo_directories(output_dir):
    """
    创建 YOLO 数据集目录结构
    """
    dirs = {
        'images_train': os.path.join(output_dir, 'images', 'train'),
        'images_val': os.path.join(output_dir, 'images', 'val'),
        'images_test': os.path.join(output_dir, 'images', 'test'),
        'labels_train': os.path.join(output_dir, 'labels', 'train'),
        'labels_val': os.path.join(output_dir, 'labels', 'val'),
        'labels_test': os.path.join(output_dir, 'labels', 'test'),
        'coco': os.path.join(output_dir, 'coco'),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


def copy_images_for_split(split_samples, dirs, source_image_dir, split_name):
    """
    为指定划分复制图像文件
    """
    if split_name == 'train':
        img_dir = dirs['images_train']
        label_dir = dirs['labels_train']
    elif split_name == 'val':
        img_dir = dirs['images_val']
        label_dir = dirs['labels_val']
    elif split_name == 'test':
        img_dir = dirs['images_test']
        label_dir = dirs['labels_test']
    else:
        raise ValueError(f"未知的划分名称: {split_name}")

    copied = 0
    missing = 0

    for sample in split_samples:
        image_name = sample['image_name']
        src_path = os.path.join(source_image_dir, image_name)

        # 如果直接路径找不到，尝试从标注文件所在目录查找
        if not os.path.exists(src_path):
            src_path = os.path.join(os.path.dirname(sample.get('asset_file', '')), image_name)

        dst_path = os.path.join(img_dir, image_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            missing += 1

    return copied, missing


# ============================================================
# 简化版 JSON（用于快速查看）
# ============================================================

def save_simple_json(samples, output_path):
    """
    保存简化版 JSON（便于人工检查）
    """
    simple_data = []

    for sample in samples:
        item = {
            'image': sample['image_name'],
            'width': sample['width'],
            'height': sample['height'],
            'keypoints': {}
        }

        for kp_name, kp in sample['keypoints'].items():
            item['keypoints'][kp_name] = {
                'x': kp['x'],
                'y': kp['y']
            }

        simple_data.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)

    print(f"简化版 JSON 已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("VoTT -> COCO + YOLO (YOLOv8-Pose) 转换工具")
    print("=" * 60)
    print()

    # 1. 解析 VoTT 项目
    print("[1/7] 解析 VoTT 项目...")
    target_folder, source_folder, tags = parse_vott_project(VOTT_FILE)
    print()

    # 2. 查找所有标注文件
    print("[2/7] 查找标注文件...")
    asset_files = find_asset_files(target_folder)
    print()

    # 3. 解析所有标注
    print("[3/7] 解析标注数据...")
    all_samples = []
    all_warnings = []
    tag_stats = defaultdict(int)
    tag_mapped_stats = defaultdict(int)

    for asset_file in asset_files:
        try:
            image_name, width, height, keypoints_dict, warnings = parse_asset_file(asset_file)

            if warnings:
                all_warnings.extend(warnings)

            if not keypoints_dict:
                continue

            # 统计
            for raw_tag in keypoints_dict.keys():
                tag_mapped_stats[raw_tag] += 1

            sample = {
                'image_name': image_name,
                'width': width,
                'height': height,
                'keypoints': keypoints_dict,
                'asset_file': asset_file
            }

            all_samples.append(sample)

        except Exception as e:
            print(f"  处理 {asset_file} 出错: {e}")
            continue

    print(f"成功解析: {len(all_samples)} 个有效样本")
    if all_warnings:
        print(f"警告: {len(all_warnings)} 条（仅显示前10条）")
        for w in all_warnings[:10]:
            print(f"  - {w}")
    print()

    # 4. 统计关键点映射情况
    print("[4/7] 关键点映射统计:")
    for kp_name in KEYPOINT_NAMES:
        count = sum(1 for s in all_samples if kp_name in s['keypoints'])
        pct = count / len(all_samples) * 100 if all_samples else 0
        marker = "✓" if pct > 80 else "⚠" if pct > 50 else "✗"
        print(f"  {marker} {kp_name}: {count}/{len(all_samples)} ({pct:.1f}%)")

    # 检查是否有未映射的标签
    all_raw_tags = set()
    for asset_file in asset_files:
        try:
            with open(asset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for region in data.get('regions', []):
                for tag in region.get('tags', []):
                    all_raw_tags.add(tag)
        except:
            pass

    print()
    print("VoTT 中的原始标签:")
    for tag in sorted(all_raw_tags):
        mapped = map_label(tag)
        status = f"-> {mapped}" if mapped else "(已忽略)"
        print(f"  '{tag}' {status}")
    print()

    # 5. 划分数据集
    print("[5/7] 划分数据集...")
    splits = split_dataset(all_samples, SPLIT_RATIOS, RANDOM_SEED)
    for split_name, split_data in splits.items():
        if split_data:
            print(f"  {split_name}: {len(split_data)} 张")
    print()

    # 6. 创建输出目录结构
    print("[6/7] 生成输出文件...")
    dirs = setup_yolo_directories(OUTPUT_DIR)

    # 复制图像到对应目录
    image_source = source_folder if source_folder else target_folder
    print(f"  图像源目录: {image_source}")

    for split_name in ['train', 'val', 'test']:
        split_data = splits[split_name]
        if not split_data:
            continue

        copied, missing = copy_images_for_split(split_data, dirs, image_source, split_name)
        print(f"  {split_name}: 复制 {copied} 张图像, 缺失 {missing} 张")

        # 保存 YOLO 标签
        if split_name == 'train':
            label_dir = dirs['labels_train']
        elif split_name == 'val':
            label_dir = dirs['labels_val']
        else:
            label_dir = dirs['labels_test']

        save_yolo_labels(split_data, label_dir)

    # 保存 COCO 格式（合并所有数据）
    print()
    coco_data = build_coco_data(all_samples)
    coco_path = os.path.join(dirs['coco'], 'annotations.json')
    save_coco_json(coco_data, coco_path)

    # 保存划分后的 COCO
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        split_coco = build_coco_data(split_data)
        split_coco_path = os.path.join(dirs['coco'], f'{split_name}.json')
        save_coco_json(split_coco, split_coco_path)

    # 保存简化版 JSON
    simple_path = os.path.join(OUTPUT_DIR, 'goalkeeper_simple.json')
    save_simple_json(all_samples, simple_path)

    # 保存 YOLO data.yaml
    yaml_path = create_yolo_yaml(OUTPUT_DIR, CLASS_NAME, len(KEYPOINT_NAMES))

    print()

    # 7. 最终统计
    print("[7/7] 转换完成!")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    print("生成的文件:")
    print(f"  COCO 格式:")
    print(f"    - {dirs['coco']}/annotations.json (全部数据)")
    for split_name in ['train', 'val', 'test']:
        if splits[split_name]:
            print(f"    - {dirs['coco']}/{split_name}.json")
    print()
    print(f"  YOLO 格式:")
    print(f"    - {yaml_path} (数据集配置)")
    print(f"    - {dirs['images_train']}/ (训练图像)")
    print(f"    - {dirs['labels_train']}/ (训练标签)")
    print(f"    - {dirs['images_val']}/ (验证图像)")
    print(f"    - {dirs['labels_val']}/ (验证标签)")
    print()
    print(f"  其他:")
    print(f"    - {simple_path} (简化版 JSON)")
    print()
    print("=" * 60)
    print("YOLO 训练命令示例:")
    print("-" * 60)
    print("from ultralytics import YOLO")
    print("model = YOLO('yolov8n-pose.pt')  # 加载预训练模型")
    print(f"model.train(data='{yaml_path.replace(chr(92), '/')}', epochs=100, imgsz=640)")
    print("=" * 60)


if __name__ == '__main__':
    main()
