import torch
import pickle
import numpy as np
import cv2
from ultralytics import YOLO
import json
import os
from collections import deque

class VideoDivePredictor:
    """
    视频守门员扑救方向预测器
    支持逐帧分析和时序平滑
    """
    def __init__(self, pose_model_path, dive_model_dir, smoothing_frames=5):
        """
        smoothing_frames: 时序平滑窗口大小（帧数）
        """
        # 检查姿态模型
        if not os.path.exists(pose_model_path):
            print(f"⚠️ 模型不存在: {pose_model_path}")
            print("使用预训练模型: yolov8n-pose.pt")
            pose_model_path = 'yolov8n-pose.pt'

        # 加载YOLO姿态模型
        self.pose_model = YOLO(pose_model_path)
        print(f"✓ 姿态模型加载成功")

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载方向预测模型
        with open(f'{dive_model_dir}/dataset.pkl', 'rb') as f:
            data = pickle.load(f)

        self.dive_model = self._create_model(data['feature_dim'])

        # 第 48 行 - Windows 绝对路径硬编码
        model_path = f'{dive_model_dir}/C:\Users\chen3\Desktop\ok\goalkeeper_training\exp_cpu\weights\best.pt'
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except:
            state_dict = torch.load(model_path, map_location=self.device)

        self.dive_model.load_state_dict(state_dict)
        self.dive_model.eval()

        self.scaler = data['scaler']

        # 关键点映射
        self.kp_names = [
            'head', 'shoulde-left', 'shoudle-right',
            'hand-left', 'hand-right',
            'knee-left', 'knee-right',
            'foot-left', 'foot-right'
        ]

        # 时序平滑
        self.smoothing_frames = smoothing_frames
        self.direction_history = deque(maxlen=smoothing_frames)

        print("✓ 方向预测模型加载成功")
        print(f"✓ 时序平滑窗口: {smoothing_frames}帧\n")

    def _create_model(self, input_dim, hidden_dim=64):
        import torch.nn as nn
        class DiveDirectionNet(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 3)
                )
            def forward(self, x): return self.net(x)
        return DiveDirectionNet(input_dim, hidden_dim).to(self.device)

    def _compute_features(self, kps, w, h):
        """计算特征向量"""
        features = []
        coords = []
        for name in self.kp_names:
            if name in kps:
                coords.extend([kps[name]['x'] / w, kps[name]['y'] / h])
            else:
                coords.extend([0, 0])
        features.extend(coords)

        def get(name):
            return (kps[name]['x'] / w, kps[name]['y'] / h) if name in kps else None

        head, sh_l, sh_r = get('head'), get('shoulde-left'), get('shoudle-right')
        if head and sh_l and sh_r:
            features.extend([head[0] - (sh_l[0]+sh_r[0])/2, head[1] - (sh_l[1]+sh_r[1])/2])
        else:
            features.extend([0, 0])

        hand_l, hand_r = get('hand-left'), get('hand-right')
        if hand_l and hand_r and sh_l and sh_r:
            features.extend([hand_l[1]-sh_l[1], hand_r[1]-sh_r[1],
                           (hand_l[1]-sh_l[1]) - (hand_r[1]-sh_r[1])])
        else:
            features.extend([0, 0, 0])

        knee_l, knee_r, foot_l, foot_r = get('knee-left'), get('knee-right'), get('foot-left'), get('foot-right')
        if knee_l and knee_r and foot_l and foot_r:
            features.extend([knee_l[1]-foot_l[1], knee_r[1]-foot_r[1],
                          knee_l[0]-foot_l[0], knee_r[0]-foot_r[0]])
        else:
            features.extend([0, 0, 0, 0])

        valid_x = [c for i, c in enumerate(coords) if i % 2 == 0 and c > 0]
        if valid_x:
            features.append(np.mean(valid_x) - 0.5)
            features.append(max(valid_x) - min(valid_x))
        else:
            features.extend([0, 0])

        return np.array(features, dtype=np.float32)

    def predict_frame(self, frame):
        """预测单帧"""
        results = self.pose_model(frame, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return None, None

        h, w = frame.shape[:2]
        predictions = []

        for kp_data in results.keypoints.data:
            keypoints = kp_data.cpu().numpy()

            # 提取关键点
            kp_dict = {}
            for i, name in enumerate(self.kp_names):
                if i < len(keypoints):
                    x, y, conf = keypoints[i]
                    if conf > 0.3:
                        kp_dict[name] = {'x': x * w, 'y': y * h}

            # 计算特征
            features = self._compute_features(kp_dict, w, h)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)

            # 预测
            with torch.no_grad():
                output = self.dive_model(features_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = torch.argmax(output).item()

            direction = pred_class - 1
            direction_names = {-1: 'LEFT', 0: 'CENTER', 1: 'RIGHT'}

            predictions.append({
                'direction': direction,
                'direction_name': direction_names[direction],
                'confidence': float(probs[pred_class]),
                'probabilities': {
                    'left': float(probs[0]),
                    'center': float(probs[1]),
                    'right': float(probs[2])
                },
                'keypoints': kp_dict
            })

        return predictions, results

    def smooth_direction(self, current_pred):
        """时序平滑"""
        if current_pred is None:
            return None

        self.direction_history.append(current_pred['direction'])

        if len(self.direction_history) < 3:
            return current_pred

        # 投票机制
        from collections import Counter
        votes = Counter(self.direction_history)
        smoothed_direction = votes.most_common(1)[0][0]

        # 更新预测结果
        direction_names = {-1: 'LEFT', 0: 'CENTER', 1: 'RIGHT'}
        smoothed_pred = current_pred.copy()
        smoothed_pred['direction'] = smoothed_direction
        smoothed_pred['direction_name'] = direction_names[smoothed_direction]
        smoothed_pred['smoothed'] = True

        return smoothed_pred

    def draw_prediction(self, frame, prediction, frame_idx=0, fps=30):
        """在帧上绘制预测结果"""
        if prediction is None:
            cv2.putText(frame, "No Goalkeeper Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        kp = prediction['keypoints']
        direction = prediction['direction_name']
        conf = prediction['confidence']

        # 颜色映射
        color_map = {
            'head': (0, 0, 255), 'shoulde-left': (0, 255, 0), 'shoudle-right': (0, 255, 0),
            'hand-left': (0, 255, 255), 'hand-right': (0, 255, 255),
            'knee-left': (255, 0, 0), 'knee-right': (255, 0, 0),
            'foot-left': (255, 0, 255), 'foot-right': (255, 0, 255)
        }

        # 绘制关键点
        for name, coord in kp.items():
            x, y = int(coord['x']), int(coord['y'])
            cv2.circle(frame, (x, y), 5, color_map.get(name, (128, 128, 128)), -1)

        # 绘制骨骼
        skeleton = [
            ('head', 'shoulde-left'), ('head', 'shoudle-right'),
            ('shoulde-left', 'hand-left'), ('shoudle-right', 'hand-right'),
            ('shoulde-left', 'knee-left'), ('shoudle-right', 'knee-right'),
            ('knee-left', 'foot-left'), ('knee-right', 'foot-right')
        ]
        for start, end in skeleton:
            if start in kp and end in kp:
                x1, y1 = int(kp[start]['x']), int(kp[start]['y'])
                x2, y2 = int(kp[end]['x']), int(kp[end]['y'])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 显示预测结果
        text = f"{direction} ({conf:.2f})"
        text_colors = {'LEFT': (255, 0, 0), 'CENTER': (0, 255, 0), 'RIGHT': (0, 0, 255)}
        text_color = text_colors.get(direction, (255, 255, 255))

        # 主标题背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

        # 帧信息
        info_text = f"Frame: {frame_idx} | Time: {frame_idx/fps:.2f}s"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 概率条
        bar_y = 60
        for j, (name, prob) in enumerate(prediction['probabilities'].items()):
            bar_width = int(prob * 200)
            y_pos = bar_y + j * 25
            cv2.rectangle(frame, (10, y_pos), (10 + bar_width, y_pos + 20), text_color, -1)
            cv2.putText(frame, f"{name}: {prob:.2f}", (220, y_pos + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 如果经过平滑，显示标识
        if prediction.get('smoothed'):
            cv2.putText(frame, "[SMOOTHED]", (10, bar_y + 3*25 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def process_video(self, video_path, output_path=None, save_frames=False,
                     output_dir=None, show_live=False, skip_frames=1):
        """
        处理视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（.mp4）
            save_frames: 是否保存单帧图片
            output_dir: 帧保存目录
            show_live: 是否实时显示
            skip_frames: 跳帧处理（每N帧处理1帧，加速）
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {video_path}")
            return None

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {total_frames/fps:.2f}s")
        print(f"  跳帧: {skip_frames} (每{skip_frames}帧处理1帧)")

        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps/skip_frames, (width, height))
            print(f"  输出视频: {output_path}")

        # 创建输出目录
        if save_frames and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 处理循环
        frame_idx = 0
        processed_idx = 0
        results = []

        print(f"\n开始处理...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧处理
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # 预测
            predictions, _ = self.predict_frame(frame)

            if predictions and len(predictions) > 0:
                pred = predictions[0]  # 取第一个检测到的人

                # 时序平滑
                smoothed_pred = self.smooth_direction(pred)

                # 绘制结果
                display_frame = self.draw_prediction(frame.copy(), smoothed_pred,
                                                    processed_idx, fps/skip_frames)

                # 记录结果
                results.append({
                    'frame': frame_idx,
                    'processed_frame': processed_idx,
                    'time': frame_idx / fps,
                    'direction': smoothed_pred['direction_name'],
                    'confidence': smoothed_pred['confidence'],
                    'probabilities': smoothed_pred['probabilities']
                })
            else:
                # 未检测到
                display_frame = frame.copy()
                cv2.putText(display_frame, "No Detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.direction_history.clear()  # 清空历史

            # 写入视频
            if writer:
                writer.write(display_frame)

            # 保存单帧
            if save_frames and output_dir and predictions:
                frame_path = f"{output_dir}/frame_{processed_idx:05d}.jpg"
                cv2.imwrite(frame_path, display_frame)

            # 实时显示
            if show_live:
                cv2.imshow('Goalkeeper Dive Prediction', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断")
                    break

            # 进度显示
            if processed_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  进度: {progress:.1f}% ({frame_idx}/{total_frames})")

            frame_idx += 1
            processed_idx += 1

        # 释放资源
        cap.release()
        if writer:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()

        print(f"\n✓ 处理完成!")
        print(f"  处理帧数: {processed_idx}")
        print(f"  检测到守门员: {len([r for r in results if r['direction'] != 'CENTER'])}帧")

        # 统计方向变化
        if results:
            from collections import Counter
            directions = [r['direction'] for r in results]
            stats = Counter(directions)
            print(f"\n方向统计:")
            for dir_name, count in stats.items():
                print(f"  {dir_name}: {count}帧 ({count/len(results)*100:.1f}%)")

        return results

    def analyze_dive_sequence(self, results, window_size=10):
        """
        分析扑救序列，检测扑救动作的开始和结束
        """
        if not results:
            return []

        dives = []
        in_dive = False
        dive_start = 0
        current_direction = 'CENTER'

        for i, r in enumerate(results):
            direction = r['direction']

            # 检测扑救开始（从中立到左/右）
            if not in_dive and direction in ['LEFT', 'RIGHT']:
                in_dive = True
                dive_start = i
                current_direction = direction
                print(f"\n[扑救开始] 帧 {r['frame']}, 时间 {r['time']:.2f}s, 方向: {direction}")

            # 检测扑救结束（回到中立或方向改变）
            elif in_dive:
                if direction == 'CENTER' or (direction != current_direction and direction != 'CENTER'):
                    dive_duration = i - dive_start
                    if dive_duration >= 3:  # 至少持续3帧才算有效扑救
                        dives.append({
                            'start_frame': results[dive_start]['frame'],
                            'end_frame': r['frame'],
                            'start_time': results[dive_start]['time'],
                            'end_time': r['time'],
                            'duration': r['time'] - results[dive_start]['time'],
                            'direction': current_direction,
                            'confidence': np.mean([results[j]['confidence']
                                                  for j in range(dive_start, i)])
                        })
                        print(f"[扑救结束] 持续 {dives[-1]['duration']:.2f}s, 平均置信度: {dives[-1]['confidence']:.3f}")
                    in_dive = False

        return dives

if __name__ == '__main__':
    # 配置路径
    pose_model = 'yolov8n-pose.pt'  # 或你自己的模型路径
    dive_model_dir = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\dive_prediction'

    # 视频路径（修改为你的视频路径）
    video_path = r"C:\Users\chen3\Desktop\ok\pujiu5.mp4"
    output_video = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\output_video.mp4'
    output_frames_dir = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\video_frames'

    print("="*60)
    print("视频守门员扑救方向预测")
    print("="*60)

    # 检查视频是否存在
    if not os.path.exists(video_path):
        print(f"\n✗ 视频不存在: {video_path}")
        print("请修改 video_path 为你的视频路径")
        exit(1)

    # 初始化预测器
    predictor = VideoDivePredictor(pose_model, dive_model_dir, smoothing_frames=5)

    # 处理视频
    print("\n" + "="*60)
    results = predictor.process_video(
        video_path=video_path,
        output_path=output_video,      # 保存输出视频
        save_frames=True,               # 保存关键帧
        output_dir=output_frames_dir,   # 帧保存目录
        show_live=True,                 # 实时显示（按Q退出）
        skip_frames=2                   # 每2帧处理1帧（加速）
    )

    # 分析扑救序列
    if results:
        print("\n" + "="*60)
        print("扑救动作分析")
        print("="*60)
        dives = predictor.analyze_dive_sequence(results)

        if dives:
            print(f"\n共检测到 {len(dives)} 次扑救动作:")
            for i, dive in enumerate(dives, 1):
                print(f"\n扑救 {i}:")
                print(f"  方向: {dive['direction']}")
                print(f"  时间: {dive['start_time']:.2f}s - {dive['end_time']:.2f}s")
                print(f"  持续时间: {dive['duration']:.2f}s")
                print(f"  置信度: {dive['confidence']:.3f}")
        else:
            print("\n未检测到明显的扑救动作序列")

        # 保存详细结果
        result_json = r'C:\Users\chen3\Desktop\ok\datasets\goalkeeper\video_results.json'
        with open(result_json, 'w') as f:
            json.dump({
                'frame_results': results,
                'dive_sequences': dives
            }, f, indent=2)
        print(f"\n详细结果保存至: {result_json}")
