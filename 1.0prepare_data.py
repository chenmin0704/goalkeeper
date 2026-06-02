# # 1.0prepare_data.py - 一键准备数据
# import cv2
# import os
# from pathlib import Path
# import json
# import shutil
# from sklearn.model_selection import train_test_split
#
# # ========== 配置 ==========
# # 方法1：使用原始字符串（推荐）
# VIDEO_PATH = r"C:\Users\chen3\Desktop\ok\pujiu1.mp4"  # 注意前面的 r
#
# # 方法2：或者使用双反斜杠
# # VIDEO_PATH = "C:\\Users\\chen3\\Desktop\\ok\\pujiu5.mp4"
#
# # 方法3：或者使用正斜杠（Python自动处理）
# # VIDEO_PATH = "C:/Users/chen3/Desktop/ok/pujiu5.mp4"
#
# OUTPUT_DIR = Path("./datasets/goalkeeper2")
# CLIP_DURATION = 3  # 3秒片段
# FPS = 30
#
#
# # =========================
#
# def extract_frames(video_path, output_dir):
#     """提取视频帧"""
#     # 检查视频文件是否存在
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"❌ 找不到视频文件: {video_path}\n请检查路径是否正确！")
#
#     cap = cv2.VideoCapture(video_path)
#
#     # 检查是否成功打开
#     if not cap.isOpened():
#         raise IOError(f"❌ 无法打开视频: {video_path}\n可能是格式不支持或文件损坏")
#
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     print(f"视频信息:")
#     print(f"  路径: {video_path}")
#     print(f"  总帧数: {total_frames}")
#     print(f"  FPS: {video_fps}")
#     print(f"  时长: {total_frames / video_fps:.1f}秒")
#
#     if video_fps == 0:
#         raise ValueError("❌ 无法获取视频FPS，文件可能损坏")
#
#     interval = max(1, int(video_fps / FPS))  # 确保至少为1
#     print(f"  采样间隔: 每{interval}帧取1帧")
#
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     frames = []
#     frame_id = 0
#     saved = 0
#
#     print(f"\n正在提取帧...")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         if frame_id % interval == 0:
#             # 裁剪守门员区域（中央偏右）
#             h, w = frame.shape[:2]
#
#             # 安全检查：确保裁剪区域有效
#             if h < 50 or w < 50:
#                 print(f"⚠️  帧尺寸过小: {w}x{h}，跳过")
#                 frame_id += 1
#                 continue
#
#             crop = frame[int(h * 0.1):int(h * 0.9), int(w * 0.3):int(w * 0.8)]
#
#             # 检查裁剪结果
#             if crop.size == 0:
#                 print(f"⚠️  裁剪失败，跳过帧 {frame_id}")
#                 frame_id += 1
#                 continue
#
#             # 保存
#             name = f"frame_{saved:05d}.jpg"
#             save_path = output_dir / name
#             success = cv2.imwrite(str(save_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
#
#             if success:
#                 frames.append(name)
#                 saved += 1
#
#                 if saved % 100 == 0:
#                     print(f"  ✅ 已提取 {saved} 帧")
#             else:
#                 print(f"⚠️  保存失败: {save_path}")
#
#         frame_id += 1
#
#         # 防止无限循环（调试时有用）
#         if frame_id > 100000:
#             print("⚠️  达到最大帧数限制，停止")
#             break
#
#     cap.release()
#
#     if saved == 0:
#         raise RuntimeError(f"❌ 未能提取任何帧！请检查视频文件是否有效")
#
#     print(f"\n✅ 共提取 {saved} 帧到 {output_dir}")
#     return frames
#
#
# def split_dataset(frame_names, output_dir):
#     """划分训练/验证集"""
#     if len(frame_names) == 0:
#         raise ValueError("❌ 没有帧可以划分！")
#
#     # 如果样本太少，调整划分比例
#     if len(frame_names) < 10:
#         print(f"⚠️  样本较少({len(frame_names)}个)，全部用于训练")
#         train, val = frame_names, []
#     else:
#         train, val = train_test_split(frame_names, test_size=0.2, random_state=42)
#
#     for split, names in [("train", train), ("val", val)]:
#         if len(names) == 0:
#             continue
#
#         split_dir = output_dir / split / "images"
#         split_dir.mkdir(parents=True, exist_ok=True)
#
#         for name in names:
#             src = output_dir / "all_frames" / name
#             dst = split_dir / name
#             if src.exists():
#                 shutil.copy(src, dst)
#             else:
#                 print(f"⚠️  源文件不存在: {src}")
#
#     print(f"✅ 训练集: {len(train)}, 验证集: {len(val)}")
#
#
# def create_yaml(output_dir):
#     """创建YOLO配置文件"""
#     yaml_content = f"""path: {output_dir.absolute().as_posix()}  # 使用正斜杠兼容Windows
# train: train/images
# val: val/images
#
# kpt_shape: [20, 3]  # 20个关键点, [x,y,visibility]
# flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19]
#
# names:
#   0: nose
#   1: left_eye
#   2: right_eye
#   3: left_ear
#   4: right_ear
#   5: left_shoulder
#   6: right_shoulder
#   7: left_elbow
#   8: right_elbow
#   9: left_wrist
#   10: right_wrist
#   11: left_hip
#   12: right_hip
#   13: left_knee
#   14: right_knee
#   15: left_ankle
#   16: right_ankle
#   17: left_glove
#   18: right_glove
#   19: head_top
#
# skeleton:
#   - [0, 1], [0, 2], [1, 3], [2, 4]
#   - [5, 7], [7, 9], [5, 11], [11, 13], [13, 15]
#   - [6, 8], [8, 10], [6, 12], [12, 14], [14, 16]
#   - [5, 6], [11, 12]
#   - [9, 17], [10, 18]
#   - [0, 19]
# """
#     yaml_path = output_dir / "data.yaml"
#     with open(yaml_path, "w", encoding='utf-8') as f:
#         f.write(yaml_content)
#     print(f"✅ 配置文件: {yaml_path}")
#
#
# if __name__ == "__main__":
#     try:
#         # 1. 提取所有帧
#         all_frames_dir = OUTPUT_DIR / "all_frames"
#         frames = extract_frames(VIDEO_PATH, all_frames_dir)
#
#         if len(frames) == 0:
#             raise RuntimeError("没有提取到任何帧！")
#
#         # 2. 划分数据集
#         split_dataset(frames, OUTPUT_DIR)
#
#         # 3. 创建配置
#         create_yaml(OUTPUT_DIR)
#
#         print("\n" + "=" * 50)
#         print("🎉 数据准备完成！下一步：用VoTT标注")
#         print(f"📁 标注目录: {all_frames_dir.absolute()}")
#         print(f"📊 共 {len(frames)} 张图像")
#         print("=" * 50)
#
#     except Exception as e:
#         print(f"\n❌ 错误: {e}")
#         print("\n💡 常见问题:")
#         print("   1. 视频路径是否正确？检查文件名拼写")
#         print("   2. 视频文件是否损坏？尝试用播放器打开")
#         print("   3. 路径中是否有中文或特殊字符？")
#         exit(1)


# prepare_data.py - 视频抽帧 + 统一尺寸预处理
import cv2
import os
from pathlib import Path

# ========== 配置 ==========
VIDEO_PATH = r"C:\Users\chen3\Desktop\ok\pujiu8.mp4"  # 视频路径
OUTPUT_DIR = Path("./datasets/goalkeeper6")  # 输出目录
TARGET_SIZE = (1260, 1344)  # 目标尺寸 (宽, 高)
FPS = 30  # 目标采样帧率

# 抽帧模式选择（二选一）：
# mode = "fps"     # 按帧率采样：每秒取 FPS 帧
# mode = "interval" # 按间隔采样：每 N 帧取 1 帧
mode = "fps"

# 如果 mode="interval"，设置间隔（每10帧取1帧）
INTERVAL = 10

# 是否保存原始尺寸（用于对比）
SAVE_ORIGINAL = False


# =========================


def resize_keep_aspect(frame, target_size):
    """
    等比例缩放并填充到目标尺寸
    保持原图比例，不足部分用黑色填充
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)

    # 等比例缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 创建黑色背景画布
    canvas = cv2.resize(resized, target_size)  # 先拉伸（会变形，仅创建画布）
    canvas[:] = (0, 0, 0)  # 填充黑色

    # 计算居中位置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # 将缩放后的图像居中放置
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def resize_stretch(frame, target_size):
    """直接拉伸到目标尺寸（可能变形）"""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)


def extract_frames(video_path, output_dir):
    """提取视频帧并统一尺寸"""

    # 检查视频文件
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 找不到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ 无法打开视频: {video_path}")

    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"视频信息:")
    print(f"  路径: {video_path}")
    print(f"  总帧数: {total_frames}")
    print(f"  视频FPS: {video_fps:.2f}")
    print(f"  时长: {duration:.1f}秒")
    print(f"  目标尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  抽帧模式: {mode}")

    # 确定采样间隔
    if mode == "fps":
        # 按目标帧率采样
        interval = max(1, int(video_fps / FPS))
        print(f"  采样间隔: 每{interval}帧取1帧 (目标{FPS}fps)")
    else:
        interval = INTERVAL
        print(f"  采样间隔: 每{interval}帧取1帧")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可选：创建原始尺寸目录
    if SAVE_ORIGINAL:
        original_dir = output_dir / "original"
        original_dir.mkdir(exist_ok=True)

    frames = []
    frame_id = 0
    saved = 0

    print(f"\n正在提取并预处理帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔采样
        if frame_id % interval == 0:

            # 裁剪守门员区域（中央偏右，根据你的需求调整）
            h, w = frame.shape[:2]

            if h < 50 or w < 50:
                print(f"⚠️  帧尺寸过小: {w}x{h}，跳过")
                frame_id += 1
                continue

            # 裁剪区域（可根据需要调整或注释掉）
            crop = frame[int(h * 0.1):int(h * 0.9), int(w * 0.3):int(w * 0.8)]

            if crop.size == 0:
                print(f"⚠️  裁剪失败，跳过帧 {frame_id}")
                frame_id += 1
                continue

            # 统一尺寸预处理（等比例缩放+填充）
            processed = resize_keep_aspect(crop, TARGET_SIZE)

            # 或者使用拉伸模式（可能变形但填满画面）：
            # processed = resize_stretch(crop, TARGET_SIZE)

            # 保存处理后的帧
            name = f"frame_{saved:05d}.jpg"
            save_path = output_dir / name

            success = cv2.imwrite(
                str(save_path),
                processed,
                [cv2.IMWRITE_JPEG_QUALITY, 95]  # 高质量保存
            )

            if success:
                frames.append(name)
                saved += 1

                # 可选：保存原始裁剪图
                if SAVE_ORIGINAL:
                    orig_path = original_dir / f"orig_{saved:05d}.jpg"
                    cv2.imwrite(str(orig_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

                if saved % 100 == 0:
                    print(f"  ✅ 已处理 {saved} 帧")
            else:
                print(f"⚠️  保存失败: {save_path}")

        frame_id += 1

        # 防止无限循环
        if frame_id > 1000000:
            print("⚠️  达到最大帧数限制，停止")
            break

    cap.release()

    if saved == 0:
        raise RuntimeError("❌ 未能提取任何帧！")

    print(f"\n✅ 共提取 {saved} 帧")
    print(f"   保存路径: {output_dir.absolute()}")
    print(f"   图像尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")

    return frames


def create_yaml(output_dir):
    """创建YOLO配置文件（单数据集，不划分）"""
    yaml_content = f"""path: {output_dir.absolute().as_posix()}
train: images
val: images  # 不划分，训练验证用同一套数据

kpt_shape: [20, 3]  # 20个关键点, [x,y,visibility]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19]

names:
  0: nose
  1: left_eye
  2: right_eye
  3: left_ear
  4: right_ear
  5: left_shoulder
  6: right_shoulder
  7: left_elbow
  8: right_elbow
  9: left_wrist
  10: right_wrist
  11: left_hip
  12: right_hip
  13: left_knee
  14: right_knee
  15: left_ankle
  16: right_ankle
  17: left_glove
  18: right_glove
  19: head_top

skeleton:
  - [0, 1], [0, 2], [1, 3], [2, 4]
  - [5, 7], [7, 9], [5, 11], [11, 13], [13, 15]
  - [6, 8], [8, 10], [6, 12], [12, 14], [14, 16]
  - [5, 6], [11, 12]
  - [9, 17], [10, 18]
  - [0, 19]
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"✅ 配置文件: {yaml_path}")


if __name__ == "__main__":
    try:
        # 1. 提取并预处理帧
        frames = extract_frames(VIDEO_PATH, OUTPUT_DIR)

        # 2. 创建YOLO配置
        create_yaml(OUTPUT_DIR)

        print("\n" + "=" * 50)
        print("🎉 数据准备完成！")
        print(f"📁 输出目录: {OUTPUT_DIR.absolute()}")
        print(f"📊 共 {len(frames)} 张图像")
        print(f"📐 图像尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        exit(1)