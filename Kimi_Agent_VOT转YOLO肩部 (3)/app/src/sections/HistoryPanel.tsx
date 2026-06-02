import { Clock, Trash2, Shield, Navigation } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import type { HistoryItem } from '@/types/pose';
import { POSE_TYPES } from '@/types/pose';

interface HistoryPanelProps {
  history: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  onClear: () => void;
  currentId: string | null;
}

export function HistoryPanel({ history, onSelect, onClear, currentId }: HistoryPanelProps) {
  if (history.length === 0) return null;

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-white flex items-center gap-2 text-base">
          <Clock className="w-4 h-4 text-slate-400" />
          历史记录
          <Badge variant="outline" className="text-xs text-slate-400 border-slate-600">
            {history.length}
          </Badge>
        </CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          className="text-slate-500 hover:text-red-400 h-8 px-2"
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </CardHeader>
      <CardContent className="pt-0">
        <ScrollArea className="h-[200px]">
          <div className="space-y-2">
            {history.map(item => {
              const poseConfig = POSE_TYPES.find(p => p.name === item.result.pose);
              return (
                <button
                  key={item.id}
                  onClick={() => onSelect(item)}
                  className={`
                    w-full flex items-center gap-3 p-2 rounded-lg text-left
                    transition-all duration-200
                    ${currentId === item.id
                      ? 'bg-slate-700 ring-1 ring-emerald-500/50'
                      : 'hover:bg-slate-700/50'
                    }
                  `}
                >
                  <img
                    src={item.imageUrl}
                    alt={item.result.pose}
                    className="w-12 h-12 rounded-md object-cover flex-shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <Shield
                        className="w-3.5 h-3.5 flex-shrink-0"
                        style={{ color: poseConfig?.color || '#B2BEC3' }}
                      />
                      <span className="text-sm text-white font-medium truncate">
                        {item.result.pose}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <Navigation className="w-3 h-3 text-sky-400" />
                      <span className="text-xs text-slate-400">
                        {item.result.direction} · {(item.result.pose_conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <span className="text-[10px] text-slate-600 flex-shrink-0">
                    {item.timestamp}
                  </span>
                </button>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
