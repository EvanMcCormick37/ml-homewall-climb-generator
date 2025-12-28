import { Trash } from "lucide-react";
import type { HoldMode, HoldDetail } from "@/types";

// --- Help Overlay Component ---

export function HelpOverlay() {
  return (
    <div className="absolute bottom-4 left-4 text-xs text-zinc-600 bg-zinc-900/80 px-3 py-2 rounded-lg z-10">
      <div className="flex items-center gap-4">
        <span>
          <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded">Scroll</kbd> Zoom
        </span>
        <span>
          <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded">Shift+Drag</kbd>{" "}
          Pan
        </span>
        <span>
          <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded">Ctrl+Z</kbd> Undo
        </span>
      </div>
    </div>
  );
}

// --- Pull Direction Arrow Component ---

function PullDirectionArrow({
  pullX,
  pullY,
  color,
  size = 80,
}: {
  pullX: number;
  pullY: number;
  color: string;
  size?: number;
}) {
  const centerX = size / 2;
  const centerY = size / 2;
  const radius = size / 2 - 8;
  const arrowLength = radius * 0.7;
  const endX = centerX + pullX * arrowLength;
  const endY = centerY + pullY * arrowLength;
  const angle = Math.atan2(pullY, pullX);
  const headLength = 10;

  return (
    <svg width={size} height={size} className="mx-auto">
      <circle
        cx={centerX}
        cy={centerY}
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        className="text-zinc-700"
      />
      <circle cx={centerX} cy={centerY} r={4} fill={color} />
      <line
        x1={centerX}
        y1={centerY}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        d={`M ${endX} ${endY} 
            L ${endX - headLength * Math.cos(angle - Math.PI / 6)} ${endY - headLength * Math.sin(angle - Math.PI / 6)}
            M ${endX} ${endY}
            L ${endX - headLength * Math.cos(angle + Math.PI / 6)} ${endY - headLength * Math.sin(angle + Math.PI / 6)}`}
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

// --- Hold Features Sidebar Component ---

export function HoldFeaturesSidebar({
  mode,
  selectedHold,
  isDragging,
  dragParams,
  getColor,
  onDeleteHold,
}: {
  mode: HoldMode;
  selectedHold: HoldDetail | null;
  isDragging: boolean;
  dragParams: {
    pull_x: number;
    pull_y: number;
    useability: number;
    x: number; // in feet
    y: number; // in feet
  };
  getColor: (useability: number) => string;
  onDeleteHold: () => void;
}) {
  const showDragPreview = mode === "add" && isDragging;
  const showSelectedHold = mode === "select" && selectedHold !== null;
  const isActive = showDragPreview || showSelectedHold;

  // Use drag params if dragging, otherwise use selected hold
  const useability = showDragPreview
    ? dragParams.useability
    : (selectedHold?.useability ?? 0);
  const pullX = showDragPreview
    ? dragParams.pull_x
    : (selectedHold?.pull_x ?? 0);
  const pullY = showDragPreview
    ? dragParams.pull_y
    : (selectedHold?.pull_y ?? -1);
  const coordX = showDragPreview ? dragParams.x : (selectedHold?.x ?? 0);
  const coordY = showDragPreview ? dragParams.y : (selectedHold?.y ?? 0);

  const percentage = Math.round((useability ?? 0) * 100);
  const barColor = isActive ? getColor(useability ?? 0) : "transparent";
  const bgColor = isActive ? "bg-zinc-700" : "bg-zinc-800";

  return (
    <div className="w-64 flex-shrink-0 border-l border-zinc-800 bg-zinc-900 p-5 flex flex-col gap-6 overflow-y-auto">
      <h2 className="text-xs font-bold text-zinc-500 uppercase tracking-widest">
        Hold Features{" "}
        {showDragPreview && (
          <span className="ml-2 text-emerald-500 text-[10px] animate-pulse">
            ● LIVE
          </span>
        )}
      </h2>

      {/* ID or Status */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-500">
          {showDragPreview ? "Status" : "Hold Index"}
        </span>
        <span className="text-sm font-bold text-zinc-200">
          {showDragPreview
            ? "New Hold"
            : showSelectedHold
              ? `#${selectedHold?.hold_index}`
              : "—"}
        </span>
      </div>

      {/* Useability */}
      <div className="flex flex-col gap-2">
        <span className="text-xs text-zinc-500">Useability</span>
        <div className={`h-3 rounded-full ${bgColor} overflow-hidden`}>
          <div
            className="h-full transition-all duration-75"
            style={{
              width: isActive ? `${percentage}%` : "0%",
              backgroundColor: barColor,
            }}
          />
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-zinc-600 uppercase font-bold tracking-tighter">
            Hard
          </span>
          <span
            className="text-xl font-black font-mono"
            style={{ color: isActive ? barColor : "#3f3f46" }}
          >
            {isActive ? `${percentage}%` : "—"}
          </span>
          <span className="text-[10px] text-zinc-600 uppercase font-bold tracking-tighter">
            Easy
          </span>
        </div>
      </div>

      <div className="border-t border-zinc-800/50" />

      {/* Pull Direction */}
      <div className="flex flex-col gap-3">
        <span className="text-xs text-zinc-500">Pull Direction</span>
        <div className="py-2">
          {isActive ? (
            <PullDirectionArrow
              pullX={pullX ?? 0}
              pullY={pullY ?? 0}
              color={barColor}
            />
          ) : (
            <div className="w-20 h-20 mx-auto rounded-full border-2 border-zinc-800 flex items-center justify-center text-zinc-600 text-xs">
              N/A
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-zinc-800/50" />

      {/* Coordinates (in feet) */}
      <div className="flex flex-col gap-3">
        <span className="text-xs text-zinc-500">Position (feet)</span>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-zinc-950/50 border border-zinc-800 rounded p-2">
            <span className="text-[9px] text-zinc-600 block uppercase font-bold">
              X (horizontal)
            </span>
            <span className="text-sm font-mono text-zinc-300">
              {isActive ? `${coordX.toFixed(2)} ft` : "—"}
            </span>
          </div>
          <div className="bg-zinc-950/50 border border-zinc-800 rounded p-2">
            <span className="text-[9px] text-zinc-600 block uppercase font-bold">
              Y (vertical)
            </span>
            <span className="text-sm font-mono text-zinc-300">
              {isActive ? `${coordY.toFixed(2)} ft` : "—"}
            </span>
          </div>
        </div>
      </div>

      {showSelectedHold && (
        <button
          onClick={onDeleteHold}
          className="mt-4 w-full py-3 bg-red-950/30 hover:bg-red-600 border border-red-900/50 text-red-500 hover:text-white rounded-lg font-bold text-xs transition-all flex items-center justify-center gap-2"
        >
          <Trash size={14} />
        </button>
      )}
    </div>
  );
}
