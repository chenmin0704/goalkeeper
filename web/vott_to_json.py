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
    vott_file = r"C:\Users\chen3\Desktop\ok\新建文件夹\new.vott"
    output_json = r"C:\Users\chen3\Desktop\ok\新建文件夹\goalkeeper_keypoints.json"

    convert(vott_file, output_json)
