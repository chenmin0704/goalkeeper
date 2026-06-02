# 3.train_pose.py - 训练守门员姿态估计模型（9关键点版本）
from ultralytics import YOLO
def train_goalkeeper_pose():
    """
    训练守门员姿态估计模型（CPU版本，9关键点）
    """
    # 加载预训练模型（nano版本，轻量级）
    model = YOLO('yolov8n-pose.pt')

    # 训练参数（适配CPU和9关键点）
    results = model.train(
        data=r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\yolo_dataset\dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=4,  # CPU建议小批次
        workers=2,  # CPU减少数据加载线程
        patience=20,
        device='cpu',  # 使用CPU

        # 优化器
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # 数据增强（降低计算量）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=3,
        flipud=0.0,
        fliplr=0.5,  # 水平翻转（关键，因为设置了flip_idx）
        mosaic=0.0,  # 关闭mosaic（省内存）
        mixup=0.0,   # 关闭mixup

        # 保存设置
        project='goalkeeper_training',
        name='exp_9keypoints_cpu',
        exist_ok=True,

        verbose=True,
        seed=42
    )

    print(f"训练完成！最佳模型: {results.best}")
    return results


if __name__ == '__main__':
    train_goalkeeper_pose()