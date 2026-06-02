import { useState, useCallback } from 'react';
import { Goal, Zap, Film, Image as ImageIcon, Shield, Gauge, Maximize, Crosshair } from 'lucide-react';
import { UploadSection } from '@/sections/UploadSection';
import { PoseCanvas } from '@/sections/PoseCanvas';
import { AnalysisPanel } from '@/sections/AnalysisPanel';
import { HistoryPanel } from '@/sections/HistoryPanel';
import { usePosePrediction } from '@/hooks/usePosePrediction';
import { Toaster, toast } from 'sonner';
import type { HistoryItem } from '@/types/pose';

// ============ 视频分析可视化面板 ============
const VideoAnalysisVisual: React.FC<{ frame: any }> = ({ frame }) => {
  if (!frame) return null;

  const { pose, pose_conf, body_angle, keypoints } = frame;

  const keypointColors = [
    { name: '头', color: 'bg-red-400', textColor: 'text-red-400' },
    { name: '左肩', color: 'bg-emerald-400', textColor: 'text-emerald-400' },
    { name: '右肩', color: 'bg-emerald-400', textColor: 'text-emerald-400' },
    { name: '左手', color: 'bg-cyan-400', textColor: 'text-cyan-400' },
    { name: '右手', color: 'bg-cyan-400', textColor: 'text-cyan-400' },
    { name: '左膝', color: 'bg-slate-400', textColor: 'text-slate-400' },
    { name: '右膝', color: 'bg-slate-400', textColor: 'text-slate-400' },
    { name: '左脚', color: 'bg-amber-400', textColor: 'text-amber-400' },
    { name: '右脚', color: 'bg-amber-400', textColor: 'text-amber-400' },
  ];

  return (
    <div className="space-y-4">
      {/* 预测 */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 bg-red-500/80" />
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-red-400" />
            <h3 className="text-sm font-semibold text-white">预测</h3>
          </div>
          <span className="px-2.5 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-bold border border-red-500/30">
            {(pose_conf * 100).toFixed(0)}%
          </span>
        </div>
        <div className="text-2xl font-bold text-white mb-1">{pose}</div>
        <div className="text-sm text-slate-400">
          {pose.includes('倒地') ? '大幅向' + (pose.includes('右') ? '右' : '左') + '倒地' :
           pose.includes('侧扑') ? '身体向' + (pose.includes('右') ? '右' : '左') + '侧倾斜扑救' :
           pose.includes('高空') ? '双手高举准备接球' : '常规站立准备姿态'}
        </div>
        <div className="mt-4">
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>置信度</span>
            <span>{(pose_conf * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-700/50 rounded-full h-2">
            <div className="bg-red-500 h-2 rounded-full transition-all" style={{ width: `${pose_conf * 100}%` }} />
          </div>
        </div>
      </div>

      {/* 身体角度 */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Gauge className="w-5 h-5 text-amber-400" />
          <h3 className="text-sm font-semibold text-white">身体角度</h3>
        </div>
        <div className="flex justify-center py-2">
          <div className="relative w-40 h-20 overflow-hidden">
            <div className="absolute bottom-0 left-0 right-0 h-40 rounded-full border-[8px] border-slate-700" />
            <div
              className="absolute bottom-0 left-0 right-0 h-40 rounded-full border-[8px] border-emerald-500 transition-all"
              style={{
                clipPath: `polygon(0 0, 100% 0, 100% ${50 - (Math.abs(body_angle) / 90) * 50}%, 0 ${50 - (Math.abs(body_angle) / 90) * 50}%)`,
                transform: body_angle < 0 ? 'scaleX(-1)' : 'none'
              }}
            />
            <div className="absolute bottom-0 left-0 text-[10px] text-slate-600 -translate-y-1">-75</div>
            <div className="absolute bottom-0 right-0 text-[10px] text-slate-600 -translate-y-1">+75</div>
            <div
              className="absolute bottom-0 left-1/2 w-0.5 h-16 bg-white origin-bottom transition-transform duration-500"
              style={{ transform: `translateX(-50%) rotate(${(body_angle / 75) * 45}deg)` }}
            />
            <div className="absolute bottom-0 left-1/2 w-2 h-2 bg-white rounded-full -translate-x-1/2" />
          </div>
        </div>
        <p className="text-center text-sm text-slate-400 mt-2">
          躯干倾斜：<span className="text-white font-medium">{body_angle.toFixed(1)}度</span>
          {Math.abs(body_angle) > 20 && <span className="text-red-400 ml-1">(扑救动作)</span>}
        </p>
      </div>

      {/* 覆盖范围 */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Maximize className="w-5 h-5 text-emerald-400" />
          <h3 className="text-sm font-semibold text-white">覆盖范围</h3>
        </div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-lg font-bold text-white">
            {Math.abs(body_angle) > 45 ? '大幅伸展' : Math.abs(body_angle) > 20 ? '中等范围' : '紧凑姿态'}
          </span>
          <span className="text-xl font-bold text-emerald-400">{Math.min(Math.abs(body_angle) * 0.4, 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-slate-700/50 rounded-full h-3 mb-2">
          <div
            className="bg-emerald-500 h-3 rounded-full transition-all"
            style={{ width: `${Math.min(Math.abs(body_angle) * 0.4, 100)}%` }}
          />
        </div>
        <p className="text-xs text-slate-500">关键点覆盖的图像面积比例</p>
      </div>

      {/* 关键点置信度 */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Crosshair className="w-5 h-5 text-purple-400" />
          <h3 className="text-sm font-semibold text-white">关键点置信度</h3>
        </div>
        <div className="space-y-3">
          {keypoints?.map((kp: any, idx: number) => {
            const colorInfo = keypointColors[idx] || { color: 'bg-slate-400', textColor: 'text-slate-400' };
            return (
              <div key={kp.name} className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${colorInfo.color}`} />
                <span className="text-sm text-slate-300 w-12">{kp.name.replace(/-(left|right)/, '').replace('shoulder', '肩').replace('hand', '手').replace('knee', '膝').replace('foot', '脚').replace('head', '头')}</span>
                <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                  <div
                    className={`${colorInfo.color} h-2 rounded-full transition-all`}
                    style={{ width: `${(kp.conf * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className={`text-sm font-medium ${colorInfo.textColor} w-10 text-right`}>
                  {(kp.conf * 100).toFixed(0)}%
                </span>
              </div>
            );
          }) || <p className="text-xs text-slate-500">关键点数据不可用</p>}
        </div>
      </div>
    </div>
  );
};

// ============ 视频预测面板 ============
const VideoPredictPanel: React.FC<{ apiUrl: string }> = ({ apiUrl }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [progress, setProgress] = useState(0);

  const handlePredict = async () => {
    if (!file) {
      toast.error('请先选择视频文件');
      return;
    }

    setLoading(true);
    setProgress(0);
    const formData = new FormData();
    formData.append('video', file);

    const interval = setInterval(() => {
      setProgress(p => Math.min(p + 3, 85));
    }, 400);

    try {
      const res = await fetch(`${apiUrl}/api/predict_video`, {
        method: 'POST',
        body: formData,
      });
      clearInterval(interval);
      setProgress(100);

      if (!res.ok) throw new Error('预测失败');
      const data = await res.json();
      setResult(data);
      toast.success(`视频分析完成！共分析 ${data.analyzed_frames} 帧`);
    } catch (err: any) {
      toast.error('视频分析失败', { description: err.message });
    } finally {
      setLoading(false);
    }
  };

  const getPoseColor = (pose: string) => {
    if (pose.includes('倒地')) return 'bg-red-500/20 text-red-400 border-red-500/30';
    if (pose.includes('侧扑')) return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    if (pose.includes('高空')) return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
  };

  const getKeyFrame = () => {
    if (!result?.frames?.length) return null;
    const diveFrame = result.frames.find((f: any) => f.pose.includes('倒地') || f.pose.includes('侧扑'));
    return diveFrame || result.frames[result.frames.length - 1];
  };

  return (
    <div className="space-y-4">
      {/* 上传区域 */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
        <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-emerald-500/50 transition-colors">
          <input
            type="file"
            accept="video/*"
            onChange={e => {
              const f = e.target.files?.[0];
              if (f) {
                if (!f.type.startsWith('video/')) {
                  toast.error('请选择视频文件');
                  return;
                }
                setFile(f);
                setResult(null);
              }
            }}
            className="hidden"
            id="video-upload"
          />
          <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center">
            <Film className="w-10 h-10 text-slate-500 mb-3" />
            <span className="text-slate-300 font-medium">{file ? file.name : '点击选择视频文件'}</span>
            <span className="text-slate-500 text-sm mt-1">MP4, AVI, MOV</span>
          </label>
        </div>

        {file && (
          <video
            src={URL.createObjectURL(file)}
            controls
            className="w-full mt-4 rounded-lg max-h-48 object-contain bg-black"
          />
        )}

        <button
          onClick={handlePredict}
          disabled={loading || !file}
          className={`mt-4 w-full py-2.5 rounded-lg font-medium transition-all ${
            loading || !file
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white'
          }`}
        >
          {loading ? `分析中 ${progress}%...` : '🎬 开始视频分析'}
        </button>
      </div>

      {/* 进度条 */}
      {loading && (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div className="bg-emerald-500 h-2 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
          </div>
          <p className="text-xs text-slate-500 mt-2">正在逐帧分析守门员姿态...</p>
        </div>
      )}

      {/* 结果展示 */}
      {result && (
        <div className="space-y-4">
          {/* 统计卡片 */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: '总帧数', value: result.total_frames, color: 'text-blue-400' },
              { label: '分析帧数', value: result.analyzed_frames, color: 'text-emerald-400' },
              { label: '帧率', value: `${result.fps} FPS`, color: 'text-purple-400' },
              { label: '姿态种类', value: Object.keys(result.pose_distribution).length, color: 'text-amber-400' },
            ].map((item) => (
              <div key={item.label} className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3 text-center">
                <div className={`text-xl font-bold ${item.color}`}>{item.value}</div>
                <div className="text-xs text-slate-500 mt-1">{item.label}</div>
              </div>
            ))}
          </div>

          {/* 关键帧可视化分析 */}
          {getKeyFrame() && (
            <div className="bg-slate-800/30 border border-slate-700/30 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-emerald-400" />
                关键帧分析结果（第{getKeyFrame().frame}帧 / {getKeyFrame().time.toFixed(2)}s）
              </h3>
              <VideoAnalysisVisual frame={getKeyFrame()} />
            </div>
          )}

          {/* 姿态分布 */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">姿态分布</h3>
            <div className="space-y-2">
              {Object.entries(result.pose_distribution).map(([pose, count]: [string, any]) => (
                <div key={pose} className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium border ${getPoseColor(pose)}`}>
                    {pose}
                  </span>
                  <div className="flex-1 bg-slate-700 rounded-full h-5 relative overflow-hidden">
                    <div className="bg-emerald-500/60 h-full rounded-full" style={{ width: `${(count / result.analyzed_frames) * 100}%` }} />
                    <span className="absolute inset-0 flex items-center justify-center text-[10px] text-slate-300">
                      {count}帧 ({((count / result.analyzed_frames) * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 逐帧表格 */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">逐帧结果</h3>
            <div className="overflow-x-auto max-h-64 overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="bg-slate-700/50 sticky top-0">
                  <tr>
                    <th className="px-2 py-2 text-left text-slate-400">帧</th>
                    <th className="px-2 py-2 text-left text-slate-400">时间</th>
                    <th className="px-2 py-2 text-left text-slate-400">姿态</th>
                    <th className="px-2 py-2 text-left text-slate-400">方向</th>
                    <th className="px-2 py-2 text-left text-slate-400">角度</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {result.frames.map((f: any) => (
                    <tr key={f.frame} className="hover:bg-slate-700/30">
                      <td className="px-2 py-2 text-slate-400 font-mono">{f.frame}</td>
                      <td className="px-2 py-2 text-slate-400">{f.time.toFixed(2)}s</td>
                      <td className="px-2 py-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] border ${getPoseColor(f.pose)}`}>
                          {f.pose}
                        </span>
                      </td>
                      <td className="px-2 py-2 text-slate-400">{f.direction}</td>
                      <td className="px-2 py-2 text-slate-400 font-mono">{f.body_angle}°</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 导出按钮 */}
          <div className="flex gap-3">
            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `video_prediction_${Date.now()}.json`;
                a.click();
                toast.success('结果已导出');
              }}
              className="flex-1 bg-emerald-600/20 border border-emerald-500/30 text-emerald-400 py-2 rounded-lg text-sm font-medium hover:bg-emerald-600/30 transition-colors"
            >
              📥 导出 JSON
            </button>
            <button
              onClick={() => { setFile(null); setResult(null); setProgress(0); }}
              className="flex-1 bg-slate-700 text-slate-300 py-2 rounded-lg text-sm font-medium hover:bg-slate-600 transition-colors"
            >
              🔄 重新上传
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ============ 主应用 ============
export default function App() {
  const {
    isLoading,
    currentResult,
    history,
    apiUrl,
    setApiUrl,
    useMock,
    setUseMock,
    predict,
    clearHistory,
  } = usePosePrediction();

  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'image' | 'video'>('image');

  const handleImageUpload = useCallback(async (file: File) => {
    try {
      const url = URL.createObjectURL(file);
      setCurrentImageUrl(url);
      setSelectedHistoryId(null);
      const result = await predict(file);
      toast.success('分析完成', { description: `检测到姿态: ${result.pose}` });
    } catch (error) {
      toast.error('分析失败', { description: '请检查API设置或稍后重试' });
    }
  }, [predict]);

  const handleSelectHistory = useCallback((item: HistoryItem) => {
    setCurrentImageUrl(item.imageUrl);
    setSelectedHistoryId(item.id);
  }, []);

  const displayResult = selectedHistoryId
    ? history.find(h => h.id === selectedHistoryId)?.result || currentResult
    : currentResult;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 text-white">
      <Toaster position="top-right" theme="dark" />

      {/* 顶部导航 */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/20 flex items-center justify-center">
              <Goal className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">Goalkeeper Analytics</h1>
              <p className="text-[10px] text-slate-500 -mt-0.5">守门员扑救姿态分析系统</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>{useMock ? '模拟模式' : 'API模式'}</span>
            </div>
            <div className="w-px h-4 bg-slate-700" />
            <span className="text-xs text-slate-500">
              已分析 {history.length} 张
            </span>
          </div>
        </div>
      </header>

      {/* 主内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* 标签切换 */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab('image')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'image'
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            图像预测
          </button>
          <button
            onClick={() => setActiveTab('video')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'video'
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            <Film className="w-4 h-4" />
            视频预测
          </button>
        </div>

        {activeTab === 'image' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* 左侧：上传 + 图像 */}
            <div className="lg:col-span-7 space-y-4">
              <UploadSection
                onImageUpload={handleImageUpload}
                isLoading={isLoading}
                apiUrl={apiUrl}
                setApiUrl={setApiUrl}
                useMock={useMock}
                setUseMock={setUseMock}
              />

              {currentImageUrl && (
                <div className="relative">
                  <PoseCanvas
                    imageUrl={currentImageUrl}
                    result={displayResult}
                  />
                </div>
              )}

              <HistoryPanel
                history={history}
                onSelect={handleSelectHistory}
                onClear={clearHistory}
                currentId={selectedHistoryId}
              />
            </div>

            {/* 右侧：分析面板 */}
            <div className="lg:col-span-5">
              <div className="lg:sticky lg:top-24">
                <AnalysisPanel result={displayResult} />
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8">
              <VideoPredictPanel apiUrl={apiUrl} />
            </div>
            <div className="lg:col-span-4">
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">视频分析说明</h3>
                <ul className="text-xs text-slate-400 space-y-2 list-disc list-inside">
                  <li>支持 MP4、AVI、MOV 格式视频</li>
                  <li>系统自动抽帧分析（每秒5帧）</li>
                  <li>检测9个关键点：头、双肩、双手、双膝、双脚</li>
                  <li>识别姿态：站立、侧扑、倒地扑救、高空接球</li>
                  <li>关键帧可视化展示姿态、方向、角度</li>
                  <li>分析结果可导出为 JSON 格式</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 底部 */}
      <footer className="border-t border-slate-700/50 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-600">
          <p>基于 YOLOv8-Pose 的守门员关键点检测与扑救姿态分析系统</p>
          <p className="mt-1">支持9个关键点检测：头、双肩、双手、双膝、双脚</p>
        </div>
      </footer>
    </div>
  );
}