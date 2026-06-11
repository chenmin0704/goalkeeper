import { useState, useCallback } from 'react';
import type { PoseResult, HistoryItem, Keypoint } from '@/types/pose';

// 模拟数据生成器（用于演示，实际使用时替换为API调用）
function generateMockResult(): PoseResult {
  const poses = ['站立准备', '向左侧扑', '向右侧扑', '向左倒地扑救', '向右倒地扑救', '高空接球'];
  const directions = ['左侧', '正中', '右侧'];
  const pose = poses[Math.floor(Math.random() * poses.length)];
  
  const keypoints: Keypoint[] = [
    { x: 0.45 + Math.random() * 0.1, y: 0.15 + Math.random() * 0.05, conf: 0.92, name: 'head' },
    { x: 0.38 + Math.random() * 0.08, y: 0.28 + Math.random() * 0.05, conf: 0.89, name: 'shoulder-left' },
    { x: 0.52 + Math.random() * 0.08, y: 0.28 + Math.random() * 0.05, conf: 0.87, name: 'shoulder-right' },
    { x: 0.30 + Math.random() * 0.1, y: 0.42 + Math.random() * 0.08, conf: 0.85, name: 'hand-left' },
    { x: 0.58 + Math.random() * 0.1, y: 0.42 + Math.random() * 0.08, conf: 0.83, name: 'hand-right' },
    { x: 0.40 + Math.random() * 0.06, y: 0.55 + Math.random() * 0.05, conf: 0.91, name: 'knee-left' },
    { x: 0.50 + Math.random() * 0.06, y: 0.55 + Math.random() * 0.05, conf: 0.90, name: 'knee-right' },
    { x: 0.38 + Math.random() * 0.06, y: 0.72 + Math.random() * 0.05, conf: 0.88, name: 'foot-left' },
    { x: 0.52 + Math.random() * 0.06, y: 0.72 + Math.random() * 0.05, conf: 0.86, name: 'foot-right' },
  ];

  return {
    pose,
    pose_conf: 0.7 + Math.random() * 0.3,
    direction: directions[Math.floor(Math.random() * directions.length)],
    direction_conf: 0.6 + Math.random() * 0.4,
    coverage: 5 + Math.random() * 20,
    coverage_desc: Math.random() > 0.5 ? '大幅伸展' : '中等范围',
    detail: `身体倾斜${(Math.random() * 60).toFixed(1)}度，双手展开`,
    body_angle: -60 + Math.random() * 120,
    keypoints,
    bbox: [0.25, 0.1, 0.5, 0.75],
  };
}

export function usePosePrediction() {
  const [isLoading, setIsLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState<PoseResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:5000');
  const [useMock, setUseMock] = useState(true);

  // 预测函数
  const predict = useCallback(async (imageFile: File) => {
    setIsLoading(true);
    
    try {
      let result: PoseResult;
      
      if (useMock) {
        // 模拟模式：延迟后返回模拟数据
        await new Promise(resolve => setTimeout(resolve, 800));
        result = generateMockResult();
      } else {
        // 真实API模式
        const formData = new FormData();
        formData.append('image', imageFile);
        
        const response = await fetch(`${apiUrl}/predict`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('API请求失败');
        result = await response.json();
      }
      
      // 创建历史记录
      const historyItem: HistoryItem = {
        id: Date.now().toString(),
        imageUrl: URL.createObjectURL(imageFile),
        result,
        timestamp: new Date().toLocaleTimeString(),
      };
      
      setCurrentResult(result);
      setHistory(prev => [historyItem, ...prev].slice(0, 50));
      
      return result;
    } catch (error) {
      console.error('预测失败:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [apiUrl, useMock]);

  // 清除当前结果
  const clearResult = useCallback(() => {
    setCurrentResult(null);
  }, []);

  // 清除历史
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return {
    isLoading,
    currentResult,
    history,
    apiUrl,
    setApiUrl,
    useMock,
    setUseMock,
    predict,
    clearResult,
    clearHistory,
  };
}
