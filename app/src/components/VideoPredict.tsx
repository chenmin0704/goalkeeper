import React, { useState, useRef } from 'react';

interface FrameResult {
  frame: number;
  time: number;
  pose: string;
  pose_conf: number;
  direction: string;
  body_angle: number;
  keypoints: Array<{x: number; y: number; conf: number; name: string}>;
}

interface VideoResponse {
  total_frames: number;
  analyzed_frames: number;
  fps: number;
  pose_distribution: Record<string, number>;
  frames: FrameResult[];
}

const VideoPredict: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VideoResponse | null>(null);
  const [error, setError] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      if (!selected.type.startsWith('video/')) {
        setError('请选择视频文件（mp4/avi/mov）');
        return;
      }
      setFile(selected);
      setError('');
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError('请先选择视频文件');
      return;
    }

    setLoading(true);
    setError('');
    setProgress(0);

    const formData = new FormData();
    formData.append('video', file);

    try {
      // 模拟进度
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 5, 90));
      }, 500);

      const response = await fetch('http://127.0.0.1:5000/api/predict_video', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || '预测失败');
      }

      const data: VideoResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || '请求失败，请检查后端服务是否运行');
    } finally {
      setLoading(false);
    }
  };

  // 获取姿态颜色
  const getPoseColor = (pose: string) => {
    if (pose.includes('倒地')) return 'text-red-600 bg-red-50';
    if (pose.includes('侧扑')) return 'text-orange-600 bg-orange-50';
    if (pose.includes('高空')) return 'text-blue-600 bg-blue-50';
    return 'text-green-600 bg-green-50';
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">🎬 守门员视频预测</h2>

      {/* 上传区域 */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
            id="video-upload"
          />
          <label
            htmlFor="video-upload"
            className="cursor-pointer flex flex-col items-center"
          >
            <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
            </svg>
            <span className="text-gray-600 font-medium">
              {file ? file.name : '点击选择视频文件'}
            </span>
            <span className="text-gray-400 text-sm mt-1">支持 MP4, AVI, MOV 格式</span>
          </label>
        </div>

        {file && (
          <div className="mt-4">
            <video
              ref={videoRef}
              src={URL.createObjectURL(file)}
              controls
              className="w-full rounded-lg max-h-64 object-contain bg-black"
            />
          </div>
        )}

        <button
          onClick={handlePredict}
          disabled={loading || !file}
          className={`mt-4 w-full py-3 rounded-lg font-semibold text-white transition-all
            ${loading || !file
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
            }`}
        >
          {loading ? '正在分析...' : '开始预测'}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 text-red-700">
          ⚠️ {error}
        </div>
      )}

      {/* 进度条 */}
      {loading && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">分析进度</span>
            <span className="text-sm font-medium text-blue-600">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-500 mt-2">正在逐帧分析守门员姿态，请稍候...</p>
        </div>
      )}

      {/* 结果展示 */}
      {result && (
        <div className="space-y-6">
          {/* 统计概览 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">📊 分析概览</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-blue-600">{result.total_frames}</div>
                <div className="text-sm text-gray-600">总帧数</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-green-600">{result.analyzed_frames}</div>
                <div className="text-sm text-gray-600">分析帧数</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-purple-600">{result.fps}</div>
                <div className="text-sm text-gray-600">视频帧率</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {Object.keys(result.pose_distribution).length}
                </div>
                <div className="text-sm text-gray-600">姿态种类</div>
              </div>
            </div>
          </div>

          {/* 姿态分布 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">🎯 姿态分布</h3>
            <div className="space-y-3">
              {Object.entries(result.pose_distribution).map(([pose, count]) => (
                <div key={pose} className="flex items-center">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium mr-3 ${getPoseColor(pose)}`}>
                    {pose}
                  </span>
                  <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                    <div
                      className="bg-blue-500 h-full rounded-full transition-all"
                      style={{ width: `${(count / result.analyzed_frames) * 100}%` }}
                    />
                    <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-gray-700">
                      {count} 帧 ({((count / result.analyzed_frames) * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 逐帧结果 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">🎞️ 逐帧分析结果</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-gray-700">帧号</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-700">时间</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-700">姿态</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-700">置信度</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-700">方向</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-700">身体角度</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {result.frames.map((frame) => (
                    <tr key={frame.frame} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-mono text-gray-600">{frame.frame}</td>
                      <td className="px-4 py-3 text-gray-600">{frame.time.toFixed(2)}s</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getPoseColor(frame.pose)}`}>
                          {frame.pose}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-600">{(frame.pose_conf * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 text-gray-600">{frame.direction}</td>
                      <td className="px-4 py-3 font-mono text-gray-600">{frame.body_angle}°</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 导出按钮 */}
          <div className="flex gap-4">
            <button
              onClick={() => {
                const dataStr = JSON.stringify(result, null, 2);
                const blob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `prediction_${Date.now()}.json`;
                a.click();
              }}
              className="flex-1 bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
            >
              📥 导出 JSON 结果
            </button>
            <button
              onClick={() => {
                setFile(null);
                setResult(null);
                setProgress(0);
              }}
              className="flex-1 bg-gray-600 text-white py-3 rounded-lg font-semibold hover:bg-gray-700 transition-colors"
            >
              🔄 重新上传
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoPredict;