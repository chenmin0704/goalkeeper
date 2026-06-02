


import json
from pathlib import Path

file_path = Path(r"C:\Users\chen3\Desktop\ok\新建文件夹\new.vott")

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"总共有 {len(data['assets'])} 张图片")

# 统计所有标注
total_boxes = 0
for asset_id, item in data['assets'].items():
    total_boxes += len(item.get('regions', []))

print(f"总共有 {total_boxes} 个标注框")  # 应该输出75