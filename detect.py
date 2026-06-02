from ultralytics import YOLO
import cv2
import numpy as np


def draw_skeleton(image, keypoints, conf_threshold=0.3):
    """
    在图像上绘制关键点和骨骼
    """
    # 关键点名称和连接
    kp_names = ['head', 'sh-L', 'sh-R', 'hand-L', 'hand-R',
                'knee-L', 'knee-R', 'foot-L', 'foot-R']

    skeleton = [
        (0, 1), (0, 2),  # 头到肩膀
        (1, 3), (2, 4),  # 肩膀到手
        (1, 5), (2, 6),  # 肩膀到膝盖
        (5, 7), (6, 8),  # 膝盖到脚
    ]

    colors = [
        (0, 0, 255),  # head - red
        (0, 255, 0),  # shoulder - green
        (0, 255, 0),
        (0, 255, 255),  # hand - yellow
        (0, 255, 255),
        (255, 0, 0),  # knee - blue
        (255, 0, 0),
        (255, 0, 255),  # foot - purple
        (255, 0, 255),
    ]

    h, w = image.shape[:2]

    # 绘制骨骼
    for start_idx, end_idx in skeleton:
        if keypoints[start_idx][2] > conf_threshold and keypoints[end_idx][2] > conf_threshold:
            x1, y1 = int(keypoints[start_idx][0] * w), int(keypoints[start_idx][1] * h)
            x2, y2 = int(keypoints[end_idx][0] * w), int(keypoints[end_idx][1] * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 绘制关键点
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_threshold:
            px, py = int(x * w), int(y * h)
            cv2.circle(image, (px, py), 5, colors[i], -1)
            cv2.circle(image, (px, py), 5, (255, 255, 255), 1)
            cv2.putText(image, kp_names[i], (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    return image


def run_inference(model_path, image_path, output_path=None):
    """
    运行推理
    """
    # 加载模型
    model = YOLO(model_path)

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 推理
    results = model(image, verbose=False)[0]

    # 绘制结果
    if results.keypoints is not None:
        for kp in results.keypoints.data:
            # 转换为 [x, y, conf] 格式，并归一化
            keypoints = []
            for point in kp:
                x, y, conf = point.cpu().numpy()
                keypoints.append([x / image.shape[1], y / image.shape[0], conf])

            image = draw_skeleton(image, keypoints)

            # 绘制边界框
            if results.boxes is not None:
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存或显示
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"结果保存至: {output_path}")
    else:
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


if __name__ == '__main__':
    # 使用最佳模型进行推理
    model_path = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\goalkeeper_training\exp\weights\best.pt'
    test_image = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\all_frames\frame_00023.jpg'
    output_image = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\result.jpg'

    run_inference(model_path, test_image, output_image)