# 生成包含边界框和关键点连接的完整脚本
import json
import os
import cv2
import numpy as np

def draw_goalkeeper_annotation(json_path, image_dir, output_dir):
    """
    绘制守门员标注：边界框 + 关键点 + 骨骼连接
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"共有 {len(data)} 个标注对象需要处理")

    # 关键点配置（名称、颜色、半径）
    kp_config = {
        'head': {'color': (0, 0, 255), 'radius': 10},        # 红色，大头
        'shoulde-left': {'color': (0, 255, 0), 'radius': 8},  # 绿色
        'shoudle-right': {'color': (0, 255, 0), 'radius': 8}, # 绿色
        'hand-left': {'color': (0, 255, 255), 'radius': 8},   # 黄色
        'hand-right': {'color': (0, 255, 255), 'radius': 8},  # 黄色
        'knee-left': {'color': (255, 0, 0), 'radius': 8},     # 蓝色
        'knee-right': {'color': (255, 0, 0), 'radius': 8},    # 蓝色
        'foot-left': {'color': (255, 0, 255), 'radius': 8},   # 紫色
        'foot-right': {'color': (255, 0, 255), 'radius': 8},  # 紫色
    }

    # 骨骼连接配置
    skeleton = [
        # 上半身
        ('head', 'shoulde-left', (0, 255, 255), 3),      # 头到左肩，青色粗线
        ('head', 'shoudle-right', (0, 255, 255), 3),     # 头到右肩
        ('shoulde-left', 'hand-left', (0, 200, 200), 2), # 左肩到左手，浅青
        ('shoudle-right', 'hand-right', (0, 200, 200), 2),# 右肩到右手

        # 躯干（肩膀到膝盖）
        ('shoulde-left', 'knee-left', (255, 255, 0), 3), # 左肩到左膝，黄色
        ('shoudle-right', 'knee-right', (255, 255, 0), 3),# 右肩到右膝

        # 下半身
        ('knee-left', 'foot-left', (255, 100, 100), 2),  # 左膝到左脚，浅蓝
        ('knee-right', 'foot-right', (255, 100, 100), 2),# 右膝到右脚
    ]

    processed = 0
    failed = []

    for item in data:
        image_name = item['image']
        keypoints = item['keypoints']
        img_width = item['width']
        img_height = item['height']

        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"  ⚠️ 跳过: {image_name} (图片不存在)")
            failed.append(image_name)
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"  ⚠️ 跳过: {image_name} (读取失败)")
            failed.append(image_name)
            continue

        img_draw = img.copy()

        # ========== 1. 计算并绘制边界框 ==========
        # 收集所有关键点坐标（排除 gialkeeper）
        points = []
        for kp_name, kp_data in keypoints.items():
            if kp_name == 'gialkeeper':
                continue
            if isinstance(kp_data, dict) and 'x' in kp_data and 'y' in kp_data:
                points.append((int(kp_data['x']), int(kp_data['y'])))

        if len(points) >= 2:  # 至少需要2个点才能画框
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            # 计算边界框（添加一些边距）
            margin = 30
            x1 = max(0, min(xs) - margin)
            y1 = max(0, min(ys) - margin)
            x2 = min(img_width, max(xs) + margin)
            y2 = min(img_height, max(ys) + margin)

            # 绘制边界框（绿色粗框）
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 在框的左上角添加标签
            label = "Goalkeeper"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_draw, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(img_draw, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # ========== 2. 绘制骨骼线 ==========
        for start_name, end_name, color, thickness in skeleton:
            if start_name in keypoints and end_name in keypoints:
                start_pt = (int(keypoints[start_name]['x']), int(keypoints[start_name]['y']))
                end_pt = (int(keypoints[end_name]['x']), int(keypoints[end_name]['y']))
                cv2.line(img_draw, start_pt, end_pt, color, thickness)

        # ========== 3. 绘制关键点 ==========
        for kp_name, kp_data in keypoints.items():
            if kp_name == 'gialkeeper':  # 特殊标记 gialkeeper 位置
                x = int(kp_data['x'])
                y = int(kp_data['y'])
                # 画十字标记
                cv2.drawMarker(img_draw, (x, y), (0, 165, 255), cv2.MARKER_CROSS, 20, 3)
                cv2.putText(img_draw, "GK", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                continue

            if isinstance(kp_data, dict) and 'x' in kp_data and 'y' in kp_data:
                x = int(kp_data['x'])
                y = int(kp_data['y'])

                config = kp_config.get(kp_name, {'color': (128, 128, 128), 'radius': 6})
                color = config['color']
                radius = config['radius']

                # 外圈（白色）
                cv2.circle(img_draw, (x, y), radius + 2, (255, 255, 255), 2)
                # 内圈（彩色）
                cv2.circle(img_draw, (x, y), radius, color, -1)

                # 添加标签（小字体，避免遮挡）
                label = kp_name.replace('shoulde', 'sh').replace('right', 'R').replace('left', 'L')
                offset_x = 12 if x < img_width - 100 else -60
                offset_y = -12 if y > 20 else 20

                # 文字背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img_draw,
                             (x + offset_x - 2, y + offset_y - th - 2),
                             (x + offset_x + tw + 2, y + offset_y + 2),
                             (0, 0, 0), -1)
                cv2.putText(img_draw, label, (x + offset_x, y + offset_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ========== 4. 添加图例 ==========
        legend_x, legend_y = 10, 30
        cv2.putText(img_draw, "Skeleton: Head(Red)-Shoulder(Green)-Hand(Yellow)-Knee(Blue)-Foot(Purple)",
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, img_draw)
        processed += 1

        if processed % 10 == 0:
            print(f"  ✅ 已处理: {processed}/{len(data)}")

    print(f"\\n{'='*50}")
    print(f"处理完成！")
    print(f"成功: {processed} 张")
    print(f"失败: {len(failed)} 张")
    print(f"输出: {output_dir}")
    if failed:
        print(f"失败列表: {failed}")

# ==================== 运行 ====================
if __name__ == '__main__':
    json_path = r'C:\\Users\\chen3\\Desktop\\ok\\datasets\\goalkeeper\\target\\vott\\goalkeeper_keypointfinal.json'
    image_dir = r'C:\\Users\\chen3\\Desktop\\ok\\datasets\\goalkeeper\\all_frames'
    output_dir = r'C:\\Users\\chen3\\Desktop\\ok\\datasets\\goalkeeper\\labeled_frames'

    draw_goalkeeper_annotation(json_path, image_dir, output_dir)
