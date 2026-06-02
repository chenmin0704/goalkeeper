import { useState, useRef, useCallback } from 'react';
import { Upload, Image, Loader2, Settings, ToggleLeft, ToggleRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';

interface UploadSectionProps {
  onImageUpload: (file: File) => void;
  isLoading: boolean;
  apiUrl: string;
  setApiUrl: (url: string) => void;
  useMock: boolean;
  setUseMock: (v: boolean) => void;
}

export function UploadSection({
  onImageUpload,
  isLoading,
  apiUrl,
  setApiUrl,
  useMock,
  setUseMock,
}: UploadSectionProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onImageUpload(file);
  }, [onImageUpload]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="space-y-4">
      {/* 设置栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSettings(!showSettings)}
            className="text-slate-400 hover:text-white"
          >
            <Settings className="w-4 h-4 mr-1" />
            API设置
          </Button>
          <button
            onClick={() => setUseMock(!useMock)}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors"
          >
            {useMock ? <ToggleRight className="w-5 h-5 text-emerald-400" /> : <ToggleLeft className="w-5 h-5 text-slate-500" />}
            {useMock ? '模拟模式' : 'API模式'}
          </button>
        </div>
      </div>

      {showSettings && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-4 pb-3 space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-400 whitespace-nowrap">API地址:</span>
              <Input
                value={apiUrl}
                onChange={e => setApiUrl(e.target.value)}
                className="bg-slate-900 border-slate-600 text-white text-sm"
                placeholder="http://127.0.0.1:5000"
              />
            </div>
            <p className="text-xs text-slate-500">
              后端API需要运行 Flask 服务，默认端口 5000
            </p>
          </CardContent>
        </Card>
      )}

      {/* 上传区域 */}
      {!previewUrl ? (
        <div
          onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
            transition-all duration-300
            ${isDragOver
              ? 'border-emerald-400 bg-emerald-400/10 scale-[1.02]'
              : 'border-slate-600 bg-slate-800/50 hover:border-slate-400 hover:bg-slate-800'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />
          <div className="flex flex-col items-center gap-4">
            <div className={`
              w-16 h-16 rounded-full flex items-center justify-center
              ${isDragOver ? 'bg-emerald-400/20' : 'bg-slate-700/50'}
              transition-colors
            `}>
              {isDragOver ? (
                <Image className="w-8 h-8 text-emerald-400" />
              ) : (
                <Upload className="w-8 h-8 text-slate-400" />
              )}
            </div>
            <div>
              <p className="text-lg font-medium text-white mb-1">
                {isDragOver ? '释放以上传图像' : '点击或拖拽上传图像'}
              </p>
              <p className="text-sm text-slate-400">
                支持 JPG、PNG、WEBP 格式
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-700">
          <img
            src={previewUrl}
            alt="上传预览"
            className="w-full h-auto max-h-[500px] object-contain"
          />
          {isLoading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center backdrop-blur-sm">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-10 h-10 text-emerald-400 animate-spin" />
                <span className="text-white font-medium">分析中...</span>
              </div>
            </div>
          )}
          <button
            onClick={() => { setPreviewUrl(null); fileInputRef.current?.click(); }}
            className="absolute top-3 right-3 bg-black/60 hover:bg-black/80 text-white
                       px-3 py-1.5 rounded-lg text-sm backdrop-blur-sm transition-colors"
          >
            更换图片
          </button>
        </div>
      )}
    </div>
  );
}
