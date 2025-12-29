import { PlusCircle, Eraser, Hand, Trash2 } from "lucide-react";
import type {
  HoldMode,
  HoldDetail,
  EnabledFeatures,
  FeatureLabel,
} from "@/types";

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

// --- Enabled Features Menu Component ---
interface EnabledFeaturesMenuProps {
  enabledFeatures: EnabledFeatures;
  onToggle: (feature: FeatureLabel) => void;
  onClose: () => void;
}

export function EnabledFeaturesMenu({
  enabledFeatures,
  onToggle,
  onClose,
}: EnabledFeaturesMenuProps) {
  return (
    <div className="absolute top-20 left-6 bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl z-50 w-64">
      <div className="p-4 border-b border-zinc-800 flex justify-between items-center">
        <h3 className="text-sm font-bold text-zinc-300 uppercase tracking-wide">
          ENABLED HOLD FEATURES
        </h3>
        <button
          onClick={onClose}
          className="text-zinc-500 hover:text-zinc-300 text-xs"
        >
          ✕
        </button>
      </div>
      <div className="p-4 space-y-3">
        <label className="flex items-center gap-3 cursor-pointer group">
          <input
            type="checkbox"
            checked={enabledFeatures.direction}
            onChange={() => onToggle("direction")}
            className="w-4 h-4 rounded border-zinc-600 bg-zinc-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-0"
          />
          <div className="flex-1">
            <div className="text-sm font-medium text-zinc-300 group-hover:text-white">
              Direction
            </div>
            <div className="text-xs text-zinc-500">
              Pull direction (pull_x, pull_y)
            </div>
          </div>
        </label>

        <label className="flex items-center gap-3 cursor-pointer group">
          <input
            type="checkbox"
            checked={enabledFeatures.useability}
            onChange={() => onToggle("useability")}
            className="w-4 h-4 rounded border-zinc-600 bg-zinc-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-0"
          />
          <div className="flex-1">
            <div className="text-sm font-medium text-zinc-300 group-hover:text-white">
              Useability
            </div>
            <div className="text-xs text-zinc-500">Hold quality/difficulty</div>
          </div>
        </label>

        <label className="flex items-center gap-3 cursor-pointer group">
          <input
            type="checkbox"
            checked={enabledFeatures.footholds}
            onChange={() => onToggle("footholds")}
            className="w-4 h-4 rounded border-zinc-600 bg-zinc-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-0"
          />
          <div className="flex-1">
            <div className="text-sm font-medium text-zinc-300 group-hover:text-white">
              Foot Holds
            </div>
            <div className="text-xs text-zinc-500">Enable foot-only holds</div>
          </div>
        </label>
      </div>
      <div className="p-3 bg-zinc-950 border-t border-zinc-800 text-xs text-zinc-500">
        <p>
          <span className="font-semibold">Note:</span> x and y are always
          required
        </p>
      </div>
    </div>
  );
}

type HotkeysAndInstructionsProps = { enabledFeatures: EnabledFeatures };
function HotkeysAndInstructions({
  enabledFeatures,
}: HotkeysAndInstructionsProps) {
  return (
    <div className="m-4 bg-zinc-900/90 backdrop-blur-md border border-zinc-800 p-4 rounded-xl shadow-2xl">
      <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-3">
        Hotkeys
      </h3>
      <div className="space-y-2 text-[11px] leading-relaxed">
        <p className="flex items-center gap-1.5">
          <span className="text-emerald-400 font-mono font-bold border px-1">
            1
          </span>
          <PlusCircle size={14} className="text-emerald-400" />
          <span className="text-zinc-400">
            {`Click ${enabledFeatures.direction || enabledFeatures.useability ? "& drag " : ""}to create a new hold`}
          </span>
        </p>
        <p className="flex items-center gap-1.5">
          <span className="text-red-400 font-mono font-bold border px-1">
            2
          </span>
          <Eraser size={14} className="text-red-400" />
          <span className="text-zinc-400">Click on a hold to delete it</span>
        </p>
        <p className="flex items-center gap-1.5">
          <span className="text-blue-400 font-mono font-bold border px-1">
            3
          </span>
          <Hand size={14} className="text-blue-400" />
          <span className="text-zinc-400">
            Click on a hold to view hold features
          </span>
        </p>
        <div className="pt-2 border-t border-zinc-800 text-zinc-500 space-y-1">
          <p className="flex items-center gap-1.5">
            <span>•</span>
            <span className="text-zinc-400 border px-1">Shift + Drag</span>
            <span>Pan camera</span>
          </p>
          <p className="flex items-center gap-1.5">
            <span>•</span>
            <span className="text-zinc-400 border px-1">Scroll</span>
            <span>Zoom in/out</span>
          </p>
          <p className="flex items-center gap-1.5">
            <span>•</span>
            <span className="text-zinc-400 border px-1">Ctrl + Z</span>
            <span>Undo last created hold</span>
          </p>
          {enabledFeatures.footholds && (
            <p className="flex items-center gap-1.5">
              <span>•</span>
              <span className="text-zinc-400 border px-1">x</span>
              <span>Toggle add hand vs foot</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Hold Features Sidebar Component ---
interface HoldFeaturesSidebarProps {
  mode: HoldMode;
  enabledFeatures: EnabledFeatures;
  selectedHold: HoldDetail | null;
  isDragging: boolean;
  dragParams: {
    x: number;
    y: number;
    pull_x: number;
    pull_y: number;
    useability?: number;
  };
  getColor: (u: number) => string;
  onDeleteHold: () => void;
}

export function HoldFeaturesSidebar({
  mode,
  enabledFeatures,
  selectedHold,
  isDragging,
  dragParams,
  getColor,
  onDeleteHold,
}: HoldFeaturesSidebarProps) {
  // Determine what to display
  const displayHold = mode === "add" && isDragging ? dragParams : selectedHold;
  const hasDirection =
    displayHold &&
    "pull_x" in displayHold &&
    displayHold.pull_x !== undefined &&
    displayHold.pull_x !== null &&
    "pull_y" in displayHold &&
    displayHold.pull_y !== undefined &&
    displayHold.pull_y !== null;
  const hasUseability =
    displayHold &&
    "useability" in displayHold &&
    displayHold.useability !== undefined &&
    displayHold.useability !== null;

  if (!displayHold) {
    return (
      <aside className="w-80 bg-zinc-900 border-l border-zinc-800 flex items-center justify-center">
        <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
      </aside>
    );
  }

  const useability = hasUseability ? displayHold.useability! : 0.5;
  const color = getColor(useability);

  return (
    <aside className="w-80 bg-zinc-900 border-l border-zinc-800 flex flex-col overflow-auto">
      <div className="p-6 border-b border-zinc-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">
            {mode === "add" ? "New Hold" : `Hold #${selectedHold?.hold_index}`}
          </h2>
          {mode === "select" && selectedHold && (
            <button
              onClick={onDeleteHold}
              className="p-2 rounded-md hover:bg-red-600/20 text-red-500 hover:text-red-400 transition-colors"
              title="Delete hold"
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>

        {/* Position (always present) */}
        <div className="space-y-3">
          <div>
            <label className="text-xs text-zinc-500 uppercase tracking-wider">
              Position
            </label>
            <div className="mt-1 grid grid-cols-2 gap-2">
              <div className="bg-zinc-950 rounded-lg p-3">
                <div className="text-xs text-zinc-600 mb-1">X</div>
                <div className="text-lg font-mono text-zinc-300">
                  {displayHold.x.toFixed(2)} ft
                </div>
              </div>
              <div className="bg-zinc-950 rounded-lg p-3">
                <div className="text-xs text-zinc-600 mb-1">Y</div>
                <div className="text-lg font-mono text-zinc-300">
                  {displayHold.y.toFixed(2)} ft
                </div>
              </div>
            </div>
          </div>

          {/* Pull Direction Arrow (only if present) */}
          {hasDirection && (
            <PullDirectionArrow
              pullX={displayHold.pull_x!}
              pullY={displayHold.pull_y!}
              color={color}
              size={100}
            />
          )}

          {/* Useability (only if present) */}
          {hasUseability && (
            <div>
              <label className="text-xs text-zinc-500 uppercase tracking-wider">
                Useability
              </label>
              <div className="mt-2 bg-zinc-950 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl font-mono text-zinc-300">
                    {(useability * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all duration-200"
                    style={{
                      width: `${useability * 100}%`,
                      backgroundColor: color,
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hotkey Instructions */}
      <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
    </aside>
  );
}
