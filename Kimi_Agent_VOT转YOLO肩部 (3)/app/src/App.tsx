import { useState, useCallback } from 'react';
import { Goal, Zap } from 'lucide-react';
import { UploadSection } from '@/sections/UploadSection';
import { PoseCanvas } from '@/sections/PoseCanvas';
import { AnalysisPanel } from '@/sections/AnalysisPanel';
import { HistoryPanel } from '@/sections/HistoryPanel';
import { usePosePrediction } from '@/hooks/usePosePrediction';
import { Toaster, toast } from 'sonner';
import type { HistoryItem } from '@/types/pose';

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

  // 显示用的result（当前分析或历史选中项）
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

            {/* 历史记录 */}
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
