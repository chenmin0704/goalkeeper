// 姿态分析类型定义

export interface Keypoint {
  x: number;
  y: number;
  conf: number;
  name: string;
}

export interface PoseResult {
  pose: string;           // 姿态名称
  pose_conf: number;      // 姿态置信度
  direction: string;      // 扑救方向
  direction_conf: number; // 方向置信度
  coverage: number;       // 覆盖范围百分比
  coverage_desc: string;  // 覆盖描述
  detail: string;         // 详细描述
  body_angle: number;     // 身体角度
  keypoints: Keypoint[];  // 9个关键点
  bbox: [number, number, number, number]; // 边界框
}

export interface HistoryItem {
  id: string;
  imageUrl: string;
  result: PoseResult;
  timestamp: string;
}

// 骨骼连接定义 (起点索引, 终点索引)
export const SKELETON_CONNECTIONS: [number, number][] = [
  [0, 1], [0, 2],       // head -> shoulders
  [1, 3], [2, 4],       // shoulders -> hands
  [1, 5], [2, 6],       // shoulders -> knees
  [5, 7], [6, 8],       // knees -> feet
  [1, 2],               // shoulder left -> shoulder right
  [5, 6],               // knee left -> knee right
];

// 关键点名称（中英文）
export const KEYPOINT_NAMES = [
  '头', '左肩', '右肩', '左手', '右手',
  '左膝', '右膝', '左脚', '右脚'
];

export const KEYPOINT_NAMES_EN = [
  'head', 'shoulder-left', 'shoulder-right',
  'hand-left', 'hand-right', 'knee-left',
  'knee-right', 'foot-left', 'foot-right'
];

// 关键点颜色
export const KEYPOINT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#4ECDC4',
  '#45B7D1', '#45B7D1', '#96CEB4',
  '#96CEB4', '#FFEAA7', '#FFEAA7'
];

// 姿态类型配置
export const POSE_TYPES = [
  { id: 'standing', name: '站立准备', color: '#4ECDC4', desc: '守门员处于常规站立姿态' },
  { id: 'dive_left', name: '向左侧扑', color: '#E17055', desc: '身体向左侧倾斜扑救' },
  { id: 'dive_right', name: '向右侧扑', color: '#E17055', desc: '身体向右侧倾斜扑救' },
  { id: 'dive_ground_left', name: '向左倒地扑救', color: '#D63031', desc: '大幅向左倒地' },
  { id: 'dive_ground_right', name: '向右倒地扑救', color: '#D63031', desc: '大幅向右倒地' },
  { id: 'catch_high', name: '高空接球', color: '#00B894', desc: '双手高举接球' },
  { id: 'unknown', name: '无法判断', color: '#B2BEC3', desc: '关键点不足或姿态不明' },
];
