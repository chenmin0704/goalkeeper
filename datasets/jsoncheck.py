import json
from pathlib import Path

json_path = Path(r"C:\Users\chen3\Desktop\ok\datasets\goalkeeper\target\vott-json-export\goalkeeper-detection-export.json")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看第一个资产的regions结构
first_asset = list(data['assets'].values())[0]
first_region = first_asset['regions'][0]

print("=== Region 完整结构 ===")
print(json.dumps(first_region, indent=2, ensure_ascii=False))