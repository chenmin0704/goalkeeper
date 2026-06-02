"""
YOLOv8-Pose 守门员关键点检测训练脚本
使用 Ultralytics 官方库
"""

from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = r"C:\Users\chen3\Desktop\ok\新建文件夹\yolo_pose_dataset\data.yaml"
MODEL = "yolov8n-pose.pt"  # 可选: n/s/m/l，n最快，l最准
EPOCHS = 100
IMGSZ = 640
BATCH = 16
NAME = "goalkeeper_pose"
PROJECT = r"C:\Users\chen3\Desktop\ok\新建文件夹\runs"


# =============================================

def train():
    """
    训练YOLOv8-Pose模型
    """
    # 加载预训练模型
    model = YOLO(MODEL)

    # 开始训练
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=20,  # 早停耐心值
        save=True,  # 保存最佳模型
        pretrained=True,  # 使用预训练权重
        optimizer="AdamW",  # 优化器
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,  # 预热epoch数
        box=7.5,  # 边框损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # DFL损失权重
        pose=12.0,  # 关键点损失权重（重要！）
        kobj=1.0,  # 关键点目标性权重
        augment=True,  # 数据增强
        mosaic=1.0,  # Mosaic增强
        flipud=0.0,  # 上下翻转（守门员通常不上下翻）
        fliplr=0.5,  # 左右翻转概率
        degrees=10,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=2,  # 剪切
        perspective=0.0,  # 透视变换
        hsv_h=0.015,  # HSV色调
        hsv_s=0.7,  # HSV饱和度
        hsv_v=0.4,  # HSV亮度
        exist_ok=False,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳模型: {PROJECT}/{NAME}/weights/best.pt")
    print("=" * 60)

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv8-Pose 守门员关键点检测")
    print("=" * 60)
    print(f"\n模型: {MODEL}")
    print(f"数据: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"图像尺寸: {IMGSZ}")
    print(f"Batch: {BATCH}")
    print()

    train()