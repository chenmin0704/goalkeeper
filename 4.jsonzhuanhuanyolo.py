# import json
# import os
# import glob
#
#
# def parse_vott_project(vott_file_path):
#     """
#     解析 .vott 项目文件
#     """
#     with open(vott_file_path, 'r', encoding='utf-8') as f:
#         project = json.load(f)
#
#     print(f"项目名: {project.get('name', 'Unknown')}")
#
#     # 获取 target 连接路径
#     target_connection = project.get('targetConnection', {})
#     provider_options = target_connection.get('providerOptions', {})
#     target_folder = provider_options.get('folderPath', '')
#
#     # 获取 source 连接路径
#     source_connection = project.get('sourceConnection', {})
#     source_options = source_connection.get('providerOptions', {})
#     source_folder = source_options.get('folderPath', '')
#
#     print(f"源图像文件夹: {source_folder}")
#     print(f"标注文件夹: {target_folder}")
#     print(f"标签列表: {[tag['name'] for tag in project.get('tags', [])]}")
#
#     return target_folder, project.get('tags', [])
#
#
# def convert_vott_to_coco(vott_file, output_json):
#     """
#     转换为 COCO 格式
#     """
#     print(f"正在处理: {vott_file}")
#     print("-" * 50)
#
#     # 解析项目文件
#     target_folder, tags = parse_vott_project(vott_file)
#
#     # 如果 target_folder 无效，使用 .vott 所在目录
#     if not target_folder or not os.path.exists(target_folder):
#         vott_dir = os.path.dirname(vott_file)
#         print(f"使用 .vott 所在目录: {vott_dir}")
#         target_folder = vott_dir
#
#     # 查找所有 asset.json 文件
#     asset_files = glob.glob(os.path.join(target_folder, '*-asset.json'))
#     if not asset_files:
#         # 尝试查找所有 json 文件（排除 .vott）
#         all_json = glob.glob(os.path.join(target_folder, '*.json'))
#         asset_files = [f for f in all_json if not f.endswith('.vott')]
#
#     if not asset_files:
#         print("错误: 找不到标注文件，请确保已导出 VoTT 项目")
#         return False
#
#     print(f"找到 {len(asset_files)} 个标注文件")
#
#     # 定义你的 9 个关键点（严格按照你的命名）
#     keypoint_names = [
#         'head',
#         'shoulde-left',  # 注意：你的拼写是 shoulde
#         'shoudle-right',  # 注意：你的拼写是 shoudle
#         'hand-left',
#         'hand-right',
#         'knee-left',
#         'knee-right',
#         'foot-left',
#         'foot-right'
#     ]
#
#     # 骨骼连接（根据你的关键点定义）
#     skeleton = [
#         [0, 1], [0, 2],  # 头连接左右肩
#         [1, 3], [2, 4],  # 肩连接手（跳过了肘部）
#         [1, 5], [2, 6],  # 肩连接膝盖（躯干）
#         [5, 7], [6, 8]  # 膝盖连接脚
#     ]
#
#     coco_data = {
#         'info': {
#             'description': 'Goalkeeper Keypoint Dataset',
#             'version': '1.0',
#             'year': 2024
#         },
#         'images': [],
#         'annotations': [],
#         'categories': [{
#             'id': 1,
#             'name': 'goalkeeper',
#             'keypoints': keypoint_names,
#             'skeleton': skeleton
#         }]
#     }
#
#     annotation_id = 1
#
#     for idx, asset_file in enumerate(sorted(asset_files)):
#         try:
#             with open(asset_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#
#             asset = data.get('asset', {})
#             image_name = asset.get('name', '')
#             width = asset.get('size', {}).get('width', 0)
#             height = asset.get('size', {}).get('height', 0)
#
#             # 收集关键点
#             keypoints_dict = {}
#             for region in data.get('regions', []):
#                 tags_list = region.get('tags', [])
#                 if not tags_list:
#                     continue
#
#                 tag = tags_list[0]
#                 bbox = region.get('boundingBox', {})
#                 if not bbox:
#                     continue
#
#                 # 计算中心点
#                 cx = bbox.get('left', 0) + bbox.get('width', 0) / 2
#                 cy = bbox.get('top', 0) + bbox.get('height', 0) / 2
#
#                 keypoints_dict[tag] = {
#                     'x': round(cx, 2),
#                     'y': round(cy, 2),
#                     'visibility': 2  # 2=可见且标注
#                 }
#
#             if not keypoints_dict:
#                 continue
#
#             # 添加图像
#             image_id = idx + 1
#             coco_data['images'].append({
#                 'id': image_id,
#                 'file_name': image_name,
#                 'width': width,
#                 'height': height
#             })
#
#             # 构建关键点数组 [x, y, visibility, x, y, visibility, ...]
#             keypoints_list = []
#             num_keypoints = 0
#
#             for kp_name in keypoint_names:
#                 if kp_name in keypoints_dict:
#                     kp = keypoints_dict[kp_name]
#                     keypoints_list.extend([kp['x'], kp['y'], kp['visibility']])
#                     num_keypoints += 1
#                 else:
#                     # 缺失的关键点
#                     keypoints_list.extend([0, 0, 0])
#
#             # 计算边界框
#             visible_kps = [keypoints_dict[k] for k in keypoints_dict]
#             xs = [kp['x'] for kp in visible_kps]
#             ys = [kp['y'] for kp in visible_kps]
#             bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
#
#             # 添加标注
#             coco_data['annotations'].append({
#                 'id': annotation_id,
#                 'image_id': image_id,
#                 'category_id': 1,
#                 'bbox': [round(x, 2) for x in bbox],
#                 'area': round(bbox[2] * bbox[3], 2),
#                 'keypoints': keypoints_list,
#                 'num_keypoints': num_keypoints,
#                 'iscrowd': 0
#             })
#
#             annotation_id += 1
#
#         except Exception as e:
#             print(f"  处理 {asset_file} 出错: {e}")
#             continue
#
#     # 保存
#     with open(output_json, 'w', encoding='utf-8') as f:
#         json.dump(coco_data, f, indent=2, ensure_ascii=False)
#
#     print("-" * 50)
#     print(f"✅ 完成！输出: {output_json}")
#     print(f"图像数: {len(coco_data['images'])}, 标注数: {len(coco_data['annotations'])}")
#     print(f"关键点: {keypoint_names}")
#
#     return True
#
#
# def convert_to_simple(vott_file, output_dir):
#     """
#     生成简化版 JSON
#     """
#     output_file = os.path.join(output_dir, "goalkeeper_simple.json")
#
#     with open(vott_file, 'r', encoding='utf-8') as f:
#         project = json.load(f)
#
#     target_folder = project.get('targetConnection', {}).get('providerOptions', {}).get('folderPath', '')
#     if not target_folder or not os.path.exists(target_folder):
#         target_folder = os.path.dirname(vott_file)
#
#     asset_files = glob.glob(os.path.join(target_folder, '*-asset.json'))
#     if not asset_files:
#         all_json = glob.glob(os.path.join(target_folder, '*.json'))
#         asset_files = [f for f in all_json if not f.endswith('.vott')]
#
#     simple_data = []
#
#     for asset_file in sorted(asset_files):
#         with open(asset_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         asset = data.get('asset', {})
#         regions = data.get('regions', [])
#
#         item = {
#             'image': asset.get('name', ''),
#             'width': asset.get('size', {}).get('width', 0),
#             'height': asset.get('size', {}).get('height', 0),
#             'keypoints': {}
#         }
#
#         for region in regions:
#             tags = region.get('tags', [])
#             if not tags:
#                 continue
#
#             tag = tags[0]
#             bbox = region.get('boundingBox', {})
#             if bbox:
#                 cx = bbox.get('left', 0) + bbox.get('width', 0) / 2
#                 cy = bbox.get('top', 0) + bbox.get('height', 0) / 2
#                 item['keypoints'][tag] = {
#                     'x': round(cx, 2),
#                     'y': round(cy, 2)
#                 }
#
#         if item['keypoints']:
#             simple_data.append(item)
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(simple_data, f, indent=2, ensure_ascii=False)
#
#     print(f"✅ 简化版: {output_file}")
#
#
# # ==================== 运行 ====================
#
# if __name__ == '__main__':
#     # 你的路径
#     vott_file = r"C:\Users\chen3\Desktop\ok\json\new.vott"
#     output_dir = r"C:\Users\chen3\Desktop\ok\json"
#
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 转换
#     output_coco = os.path.join(output_dir, "goalkeeper_keypoints.json")
#     success = convert_vott_to_coco(vott_file, output_coco)
#
#     if success:
#         convert_to_simple(vott_file, output_dir)
#         print("\n文件列表：")
#         print(f"1. {output_coco}")
#         print(f"2. {os.path.join(output_dir, 'goalkeeper_simple.json')}")

import json
import os
import glob


def parse_vott_project(vott_file_path):
    """解析 .vott 项目文件，获取标注文件夹路径"""
    with open(vott_file_path, 'r', encoding='utf-8') as f:
        project = json.load(f)

    # 获取 target 连接路径（标注文件所在目录）
    target_connection = project.get('targetConnection', {})
    target_folder = target_connection.get('providerOptions', {}).get('folderPath', '')

    # 如果路径无效，使用 .vott 所在目录
    if not target_folder or not os.path.exists(target_folder):
        target_folder = os.path.dirname(vott_file_path)

    return target_folder


def map_tag(tag):
    """
    标签名称映射：将 VoTT 中的各种拼写变体统一为标准名称
    """
    tag = tag.strip().lower()

    mapping = {
        'head': 'head',
        'shoulde-left': 'shoulder-left',
        'shoulde-left ': 'shoulder-left',
        'shoulder-left': 'shoulder-left',
        'shoudle-right': 'shoulder-right',
        'shoulde-right': 'shoulder-right',
        'shoulde-lright': 'shoulder-right',
        'shoulder-right': 'shoulder-right',
        'shouder-right': 'shoulder-right',
        'hand-left': 'hand-left',
        'hand-right': 'hand-right',
        'knee-left': 'knee-left',
        'knee-right': 'knee-right',
        'foot-left': 'foot-left',
        'foot-right': 'foot-right',
        'goalkeeper': 'goalkeeper',
    }

    if tag in mapping:
        return mapping[tag]

    # 模糊匹配（去除空格和连字符）
    tag_compact = tag.replace(' ', '').replace('-', '')
    for key, value in mapping.items():
        if key.replace(' ', '').replace('-', '') == tag_compact:
            return value

    return tag  # 未匹配的保持原样


def convert(vott_file, output_json):
    """
    VoTT 转换为 JSON：9个关键点 + goalkeeper
    """
    print(f"正在处理: {vott_file}")
    print("-" * 50)

    # 解析项目文件
    target_folder = parse_vott_project(vott_file)

    # 查找所有 asset.json 文件
    asset_files = glob.glob(os.path.join(target_folder, '*-asset.json'))
    if not asset_files:
        all_json = glob.glob(os.path.join(target_folder, '*.json'))
        asset_files = [f for f in all_json if not f.endswith('.vott')]

    if not asset_files:
        print("错误: 找不到标注文件")
        return

    print(f"找到 {len(asset_files)} 个标注文件")

    result = []

    for asset_file in sorted(asset_files):
        try:
            with open(asset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            asset = data.get('asset', {})
            regions = data.get('regions', [])

            item = {
                'image': asset.get('name', ''),
                'width': asset.get('size', {}).get('width', 0),
                'height': asset.get('size', {}).get('height', 0),
                'keypoints': {}
            }

            for region in regions:
                tags = region.get('tags', [])
                if not tags:
                    continue

                # 映射标签名（修复肩膀拼写问题）
                raw_tag = tags[0]
                tag = map_tag(raw_tag)

                bbox = region.get('boundingBox', {})
                if not bbox:
                    continue

                # 计算中心点
                cx = bbox.get('left', 0) + bbox.get('width', 0) / 2
                cy = bbox.get('top', 0) + bbox.get('height', 0) / 2

                item['keypoints'][tag] = {
                    'x': round(cx, 2),
                    'y': round(cy, 2)
                }

            if item['keypoints']:
                result.append(item)

        except Exception as e:
            print(f"  处理 {asset_file} 出错: {e}")
            continue

    # 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print(f"完成！输出: {output_json}")
    print(f"样本数: {len(result)}")

    # 统计各关键点出现次数
    print("关键点统计:")
    all_tags = ['head', 'shoulder-left', 'shoulder-right',
                'hand-left', 'hand-right', 'knee-left',
                'knee-right', 'foot-left', 'foot-right', 'goalkeeper']
    for tag in all_tags:
        count = sum(1 for item in result if tag in item['keypoints'])
        print(f"  {tag}: {count}")


# ==================== 运行 ====================

if __name__ == '__main__':
    # 修改为你的路径
    vott_file = r"C:\Users\chen3\Desktop\ok\json\new.vott"
    output_json = r"C:\Users\chen3\Desktop\ok\json\goalkeeper_keypoints.json"

    convert(vott_file, output_json)