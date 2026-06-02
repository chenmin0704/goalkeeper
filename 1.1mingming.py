import os
from pathlib import Path

# 配置
source_dir = r"C:\Users\chen3\Desktop\ok\datasets\goalkeeper6"  # 图片所在目录
start_number = 302  # 从 frame_00075 开始


def rename_frames(source_dir, start_number):
    """
    将 frame_00000.jpg, frame_00001.jpg... 重命名为
    frame_00075.jpg, frame_00076.jpg...
    """
    source_path = Path(source_dir)

    # 获取所有 frame_*.jpg 文件并排序
    frame_files = sorted(source_path.glob("frame_*.jpg"))

    if not frame_files:
        print("❌ 未找到 frame_*.jpg 文件")
        return

    print(f"找到 {len(frame_files)} 个文件")
    print(f"将从 frame_{start_number:05d}.jpg 开始重命名\n")

    # 需要检查是否会覆盖现有文件
    # 先收集所有新文件名
    rename_plan = []
    for i, old_file in enumerate(frame_files):
        new_number = start_number + i
        new_name = f"frame_{new_number:05d}.jpg"
        new_file = source_path / new_name

        # 检查目标文件是否已存在（且不是当前文件）
        if new_file.exists() and new_file != old_file:
            print(f"⚠️  警告: {new_name} 已存在，将被覆盖！")

        rename_plan.append((old_file, new_file, old_file.name, new_name))

    # 显示预览
    print("=" * 60)
    print("重命名预览（前5个和后5个）：")
    print("-" * 60)

    preview_items = rename_plan[:5]
    if len(rename_plan) > 10:
        preview_items += [("...", "...", "...", "...")] + rename_plan[-5:]

    for old_file, new_file, old_name, new_name in preview_items:
        if old_name == "...":
            print("  ...")
        else:
            print(f"  {old_name}  →  {new_name}")

    print("=" * 60)

    # 确认执行
    confirm = input("\n确认执行重命名？(yes/no): ")
    if confirm.lower() != 'yes':
        print("已取消")
        return

    # 执行重命名
    success_count = 0
    for old_file, new_file, old_name, new_name in rename_plan:
        try:
            # 如果新文件名已存在，先临时重命名（避免冲突）
            if new_file.exists() and new_file != old_file:
                temp_name = f"temp_{new_name}"
                new_file.rename(source_path / temp_name)
                print(f"  临时备份: {new_name} → {temp_name}")

            old_file.rename(new_file)
            success_count += 1

        except Exception as e:
            print(f"❌ 错误: {old_name} → {new_name}: {e}")

    print(f"\n✅ 完成！成功重命名 {success_count}/{len(frame_files)} 个文件")
    print(f"   范围: frame_{start_number:05d}.jpg ~ frame_{start_number + len(frame_files) - 1:05d}.jpg")


if __name__ == "__main__":
    rename_frames(source_dir, start_number)