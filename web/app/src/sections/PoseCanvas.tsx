import { useRef, useEffect, useState } from 'react';
import type { PoseResult } from '@/types/pose';
import { SKELETON_CONNECTIONS, KEYPOINT_NAMES, KEYPOINT_COLORS } from '@/types/pose';

interface PoseCanvasProps {
  imageUrl: string;
  result: PoseResult | null;
}

export function PoseCanvas({ imageUrl, result }: PoseCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgLoaded, setImgLoaded] = useState(false);

  useEffect(() => {
    if (!imageUrl || !result || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      setImgLoaded(true);
      // 设置canvas尺寸匹配容器宽度，保持比例
      const containerWidth = containerRef.current?.clientWidth || img.width;
      const scale = containerWidth / img.width;
      canvas.width = containerWidth;
      canvas.height = img.height * scale;

      // 绘制原图
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // 绘制关键点和骨骼
      drawKeypoints(ctx, result, canvas.width, canvas.height);
    };
    img.src = imageUrl;
  }, [imageUrl, result]);

  // 当没有result时只显示原图
  useEffect(() => {
    if (!imageUrl || result || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const containerWidth = containerRef.current?.clientWidth || img.width;
      const scale = containerWidth / img.width;
      canvas.width = containerWidth;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      setImgLoaded(true);
    };
    img.src = imageUrl;
  }, [imageUrl, result]);

  return (
    <div ref={containerRef} className="w-full rounded-xl overflow-hidden bg-slate-900 border border-slate-700">
      {!imgLoaded && (
        <img
          src={imageUrl}
          alt="原图"
          className="w-full h-auto object-contain"
        />
      )}
      <canvas
        ref={canvasRef}
        className={`w-full h-auto ${imgLoaded ? 'block' : 'hidden'}`}
      />
      {result && (
        <div className="absolute top-3 left-3 bg-black/70 text-white px-3 py-1.5 rounded-lg text-xs backdrop-blur-sm">
          检测到守门员
        </div>
      )}
    </div>
  );
}

function drawKeypoints(
  ctx: CanvasRenderingContext2D,
  result: PoseResult,
  canvasW: number,
  canvasH: number
) {
  const { keypoints, bbox } = result;

  // 绘制边界框
  if (bbox) {
    const [bx, by, bw, bh] = bbox;
    ctx.strokeStyle = 'rgba(78, 205, 196, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(bx * canvasW, by * canvasH, bw * canvasW, bh * canvasH);
    ctx.setLineDash([]);

    // 标签
    ctx.fillStyle = 'rgba(78, 205, 196, 0.9)';
    ctx.fillRect(bx * canvasW, by * canvasH - 24, 100, 24);
    ctx.fillStyle = '#000';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText('goalkeeper', bx * canvasW + 6, by * canvasH - 7);
  }

  // 绘制骨骼线
  ctx.lineWidth = 2.5;
  SKELETON_CONNECTIONS.forEach(([start, end]) => {
    const kp1 = keypoints[start];
    const kp2 = keypoints[end];
    if (!kp1 || !kp2) return;

    const x1 = kp1.x * canvasW;
    const y1 = kp1.y * canvasH;
    const x2 = kp2.x * canvasW;
    const y2 = kp2.y * canvasH;

    const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
    gradient.addColorStop(0, KEYPOINT_COLORS[start] + 'CC');
    gradient.addColorStop(1, KEYPOINT_COLORS[end] + 'CC');

    ctx.strokeStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  });

  // 绘制关键点
  keypoints.forEach((kp, i) => {
    const x = kp.x * canvasW;
    const y = kp.y * canvasH;
    const color = KEYPOINT_COLORS[i];

    // 外圈发光
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = color + '40';
    ctx.fill();

    // 内圈实心
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // 白色边框
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 标签
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px sans-serif';
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 4;
    ctx.fillText(KEYPOINT_NAMES[i], x + 10, y - 6);
    ctx.shadowBlur = 0;
  });
}
