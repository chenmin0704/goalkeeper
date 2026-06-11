import { Shield, Navigation, Maximize2, RotateCw, Activity, Crosshair } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import type { PoseResult } from '@/types/pose';
import { POSE_TYPES, KEYPOINT_NAMES, KEYPOINT_COLORS } from '@/types/pose';

interface AnalysisPanelProps {
  result: PoseResult | null;
}

export function AnalysisPanel({ result }: AnalysisPanelProps) {
  if (!result) {
    return (
      <div className="space-y-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="py-12 flex flex-col items-center justify-center text-slate-500">
            <Activity className="w-12 h-12 mb-4 text-slate-600" />
            <p className="text-lg font-medium">上传图像开始分析</p>
            <p className="text-sm mt-1">分析结果将在此显示</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const poseConfig = POSE_TYPES.find(p => p.name === result.pose) || POSE_TYPES[6];
  const isDive = result.pose.includes('侧扑') || result.pose.includes('倒地');

  return (
    <div className="space-y-4">
      {/* 姿态类型 */}
      <Card className="bg-slate-800/50 border-slate-700 overflow-hidden">
        <div className="h-1" style={{ backgroundColor: poseConfig.color }} />
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Shield className="w-5 h-5" style={{ color: poseConfig.color }} />
              姿态识别
            </CardTitle>
            <Badge
              variant="outline"
              className="text-xs font-bold"
              style={{ borderColor: poseConfig.color, color: poseConfig.color }}
            >
              {(result.pose_conf * 100).toFixed(0)}%
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white mb-1">
            {result.pose}
          </div>
          <p className="text-sm text-slate-400">{poseConfig.desc}</p>
          <div className="mt-3">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>置信度</span>
              <span>{(result.pose_conf * 100).toFixed(1)}%</span>
            </div>
            <Progress value={result.pose_conf * 100} className="h-2" />
          </div>
        </CardContent>
      </Card>

      {/* 扑救方向 */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Navigation className="w-5 h-5 text-sky-400" />
              扑救方向
            </CardTitle>
            <Badge variant="outline" className="text-xs text-sky-400 border-sky-400">
              {(result.direction_conf * 100).toFixed(0)}%
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <DirectionIndicator direction={result.direction} />
          <p className="text-sm text-slate-400 mt-2 text-center">
            预测扑救方向：<span className="text-white font-medium">{result.direction}</span>
          </p>
        </CardContent>
      </Card>

      {/* 身体角度 */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-white flex items-center gap-2">
            <RotateCw className="w-5 h-5 text-amber-400" />
            身体角度
          </CardTitle>
        </CardHeader>
        <CardContent>
          <AngleGauge angle={result.body_angle} />
          <p className="text-sm text-slate-400 mt-2 text-center">
            躯干倾斜：<span className="text-white font-medium">{result.body_angle.toFixed(1)}度</span>
            {isDive && <span className="text-rose-400 ml-2">(扑救动作)</span>}
          </p>
        </CardContent>
      </Card>

      {/* 覆盖范围 */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-white flex items-center gap-2">
            <Maximize2 className="w-5 h-5 text-emerald-400" />
            覆盖范围
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-2">
            <span className="text-white font-medium">{result.coverage_desc}</span>
            <span className="text-emerald-400 font-bold">{result.coverage.toFixed(1)}%</span>
          </div>
          <Progress value={result.coverage} max={30} className="h-2.5" />
          <p className="text-xs text-slate-500 mt-2">
            关键点覆盖的图像面积比例
          </p>
        </CardContent>
      </Card>

      {/* 关键点置信度 */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-white flex items-center gap-2">
            <Crosshair className="w-5 h-5 text-purple-400" />
            关键点置信度
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {result.keypoints.map((kp, i) => (
            <div key={i} className="flex items-center gap-3">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: KEYPOINT_COLORS[i] }}
              />
              <span className="text-sm text-slate-300 w-16">{KEYPOINT_NAMES[i]}</span>
              <div className="flex-1">
                <Progress value={kp.conf * 100} className="h-1.5" />
              </div>
              <span className="text-xs text-slate-400 w-12 text-right">
                {(kp.conf * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* 详细信息 */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-4">
          <p className="text-sm text-slate-400">
            <span className="text-slate-500">分析详情：</span>{result.detail}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// 方向指示器
function DirectionIndicator({ direction }: { direction: string }) {
  const rotation = direction === '左侧' ? -45 : direction === '右侧' ? 45 : 0;

  return (
    <div className="flex items-center justify-center py-3">
      <div className="relative w-32 h-32">
        {/* 圆形背景 */}
        <div className="absolute inset-0 rounded-full border-2 border-slate-600" />
        {/* 刻度线 */}
        <div className="absolute top-0 left-1/2 w-px h-3 bg-slate-500 -translate-x-1/2" />
        <div className="absolute bottom-0 left-1/2 w-px h-3 bg-slate-500 -translate-x-1/2" />
        <div className="absolute left-0 top-1/2 w-3 h-px bg-slate-500 -translate-y-1/2" />
        <div className="absolute right-0 top-1/2 w-3 h-px bg-slate-500 -translate-y-1/2" />

        {/* 方向标签 */}
        <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[10px] text-slate-500">上</span>
        <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] text-slate-500">下</span>
        <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[10px] text-slate-500">左</span>
        <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[10px] text-slate-500">右</span>

        {/* 中心点 */}
        <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-slate-500 rounded-full -translate-x-1/2 -translate-y-1/2" />

        {/* 箭头 */}
        <div
          className="absolute top-1/2 left-1/2 origin-bottom transition-transform duration-500"
          style={{ transform: `translate(-50%, -100%) rotate(${rotation}deg)` }}
        >
          <div className="w-0 h-0 border-l-[8px] border-r-[8px] border-b-[28px]
                         border-l-transparent border-r-transparent border-b-sky-400" />
        </div>
      </div>
    </div>
  );
}

// 角度仪表盘
function AngleGauge({ angle }: { angle: number }) {
  const clampedAngle = Math.max(-75, Math.min(75, angle));
  const percentage = (clampedAngle + 75) / 150 * 100;

  return (
    <div className="flex items-center justify-center py-2">
      <div className="relative w-40 h-20 overflow-hidden">
        {/* 弧形轨道 */}
        <svg viewBox="0 0 160 80" className="w-full h-full">
          <path
            d="M 10 80 A 70 70 0 0 1 150 80"
            fill="none"
            stroke="#334155"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* 填充弧 */}
          <path
            d="M 10 80 A 70 70 0 0 1 150 80"
            fill="none"
            stroke={Math.abs(angle) > 45 ? '#E17055' : Math.abs(angle) > 20 ? '#FDCB6E' : '#00B894'}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${percentage * 2.2} 220`}
            strokeDashoffset="0"
            className="transition-all duration-700"
          />
          {/* 中心线 */}
          <line x1="80" y1="80" x2="80" y2="15" stroke="#64748B" strokeWidth="1" strokeDasharray="4 2" />
          {/* 指针 */}
          <line
            x1="80"
            y1="80"
            x2={80 + Math.sin((angle * Math.PI) / 180) * 60}
            y2={80 - Math.cos((angle * Math.PI) / 180) * 60}
            stroke="#fff"
            strokeWidth="2.5"
            strokeLinecap="round"
            className="transition-all duration-500"
          />
          {/* 中心点 */}
          <circle cx="80" cy="80" r="4" fill="#fff" />
        </svg>

        {/* 刻度标签 */}
        <span className="absolute bottom-0 left-2 text-[9px] text-slate-500">-75</span>
        <span className="absolute bottom-0 left-1/2 -translate-x-1/2 text-[9px] text-slate-400">0</span>
        <span className="absolute bottom-0 right-2 text-[9px] text-slate-500">+75</span>
      </div>
    </div>
  );
}
