import { useState, useCallback, useRef } from "react";
import { PlusCircle, Eraser, Hand, Trash2, Lock, Unlock } from "lucide-react";
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
    <div
      className="absolute top-20 left-6 rounded-lg shadow-2xl z-50 w-64"
      style={{
        background: "var(--surface, #111113)",
        border: "1px solid var(--border, rgba(255,255,255,0.08))",
      }}
    >
      <div
        className="p-4 flex justify-between items-center"
        style={{ borderBottom: "1px solid var(--border, rgba(255,255,255,0.08))" }}
      >
        <h3
          className="uppercase tracking-wide"
          style={{
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            letterSpacing: "0.15em",
            color: "var(--text-muted, #71717a)",
          }}
        >
          ENABLED HOLD FEATURES
        </h3>
        <button
          onClick={onClose}
          style={{
            background: "transparent",
            border: "none",
            color: "var(--text-muted, #71717a)",
            cursor: "pointer",
            fontSize: "0.75rem",
          }}
        >
          ✕
        </button>
      </div>
      <div className="p-4 space-y-3">
        {(
          [
            {
              key: "direction" as FeatureLabel,
              label: "Direction",
              desc: "Pull direction (pull_x, pull_y)",
            },
            {
              key: "useability" as FeatureLabel,
              label: "Useability",
              desc: "Hold quality/difficulty",
            },
            {
              key: "footholds" as FeatureLabel,
              label: "Foot Holds",
              desc: "Enable foot-only holds",
            },
          ] as const
        ).map(({ key, label, desc }) => (
          <label key={key} className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={enabledFeatures[key]}
              onChange={() => onToggle(key)}
              className="w-4 h-4 rounded"
              style={{ accentColor: "var(--cyan, #06b6d4)" }}
            />
            <div className="flex-1">
              <div
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.7rem",
                  color: "var(--text-primary, #f4f4f5)",
                }}
              >
                {label}
              </div>
              <div
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.6rem",
                  color: "var(--text-muted, #71717a)",
                }}
              >
                {desc}
              </div>
            </div>
          </label>
        ))}
      </div>
      <div
        className="p-3"
        style={{
          background: "var(--bg, #09090b)",
          borderTop: "1px solid var(--border, rgba(255,255,255,0.08))",
        }}
      >
        <p
          style={{
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            color: "var(--text-muted, #71717a)",
          }}
        >
          <span style={{ color: "var(--text-primary, #f4f4f5)" }}>Note:</span> x
          and y are always required
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
    <div
      className="m-4 p-4 rounded-xl shadow-2xl"
      style={{
        background: "var(--bg, #09090b)",
        border: "1px solid var(--border, rgba(255,255,255,0.08))",
      }}
    >
      <h3
        className="uppercase tracking-widest mb-3"
        style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: "0.6rem",
          letterSpacing: "0.18em",
          color: "var(--text-muted, #71717a)",
        }}
      >
        Hotkeys
      </h3>
      <div className="space-y-2" style={{ fontSize: "11px", lineHeight: "1.6" }}>
        <p className="flex items-center gap-1.5">
          <span
            className="font-mono font-bold border px-1"
            style={{ color: "#34d399", borderColor: "rgba(52,211,153,0.3)" }}
          >
            1
          </span>
          <PlusCircle size={14} style={{ color: "#34d399" }} />
          <span style={{ color: "var(--text-muted, #71717a)" }}>
            {`Click ${enabledFeatures.direction || enabledFeatures.useability ? "& drag " : ""}to create a new hold`}
          </span>
        </p>
        <p className="flex items-center gap-1.5">
          <span
            className="font-mono font-bold border px-1"
            style={{ color: "#f87171", borderColor: "rgba(248,113,113,0.3)" }}
          >
            2
          </span>
          <Eraser size={14} style={{ color: "#f87171" }} />
          <span style={{ color: "var(--text-muted, #71717a)" }}>
            Click on a hold to delete it
          </span>
        </p>
        <p className="flex items-center gap-1.5">
          <span
            className="font-mono font-bold border px-1"
            style={{ color: "#60a5fa", borderColor: "rgba(96,165,250,0.3)" }}
          >
            3
          </span>
          <Hand size={14} style={{ color: "#60a5fa" }} />
          <span style={{ color: "var(--text-muted, #71717a)" }}>
            Click on a hold to view hold features
          </span>
        </p>
        <div
          className="pt-2 space-y-1"
          style={{
            borderTop: "1px solid var(--border, rgba(255,255,255,0.08))",
            color: "var(--text-muted, #71717a)",
          }}
        >
          <p className="flex items-center gap-1.5">
            <span>·</span>
            <span
              className="border px-1"
              style={{
                color: "var(--text-primary, #f4f4f5)",
                borderColor: "var(--border, rgba(255,255,255,0.08))",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
              }}
            >
              Shift + Drag
            </span>
            <span>Pan camera</span>
          </p>
          <p className="flex items-center gap-1.5">
            <span>·</span>
            <span
              className="border px-1"
              style={{
                color: "var(--text-primary, #f4f4f5)",
                borderColor: "var(--border, rgba(255,255,255,0.08))",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
              }}
            >
              Scroll
            </span>
            <span>Zoom in/out</span>
          </p>
          <p className="flex items-center gap-1.5">
            <span>·</span>
            <span
              className="border px-1"
              style={{
                color: "var(--text-primary, #f4f4f5)",
                borderColor: "var(--border, rgba(255,255,255,0.08))",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
              }}
            >
              Ctrl + Z
            </span>
            <span>Undo last created hold</span>
          </p>
          {enabledFeatures.footholds && (
            <p className="flex items-center gap-1.5">
              <span>·</span>
              <span
                className="border px-1"
                style={{
                  color: "var(--text-primary, #f4f4f5)",
                  borderColor: "var(--border, rgba(255,255,255,0.08))",
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.6rem",
                }}
              >
                x
              </span>
              <span>Toggle add hand vs foot</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Draggable Useability Bar Component ---
interface UseabilityBarProps {
  useability: number;
  color: string;
  isLocked: boolean;
  onLockedChange: (locked: boolean) => void;
  onUseabilityChange: (value: number) => void;
}

function UseabilityBar({
  useability,
  color,
  isLocked,
  onLockedChange,
  onUseabilityChange,
}: UseabilityBarProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleBarInteraction = useCallback(
    (clientX: number) => {
      if (!barRef.current || !isLocked) return;
      const rect = barRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const percentage = Math.max(0, Math.min(1, x / rect.width));
      onUseabilityChange(percentage);
    },
    [isLocked, onUseabilityChange]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!isLocked) return;
      setIsDragging(true);
      handleBarInteraction(e.clientX);
    },
    [isLocked, handleBarInteraction]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging || !isLocked) return;
      handleBarInteraction(e.clientX);
    },
    [isDragging, isLocked, handleBarInteraction]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  return (
    <div>
      <label
        style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: "0.6rem",
          letterSpacing: "0.15em",
          textTransform: "uppercase" as const,
          color: "var(--text-muted, #71717a)",
        }}
      >
        Useability
      </label>
      <div
        className="mt-2 rounded-lg p-3"
        style={{ background: "var(--bg, #09090b)" }}
      >
        <div className="flex items-center justify-between mb-2">
          <span
            className="font-mono"
            style={{
              fontSize: "1.5rem",
              color: "var(--text-primary, #f4f4f5)",
              fontFamily: "'Space Mono', monospace",
            }}
          >
            {(useability * 100).toFixed(0)}%
          </span>
          {isLocked && (
            <span
              className="flex items-center gap-1"
              style={{
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
                color: "#f59e0b",
              }}
            >
              <Lock size={12} />
              Manual
            </span>
          )}
        </div>
        <div
          ref={barRef}
          className={`h-2 rounded-full overflow-hidden ${isLocked ? "cursor-ew-resize" : ""}`}
          style={{ background: "var(--border, rgba(255,255,255,0.08))" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          <div
            className="h-full transition-all duration-75"
            style={{
              width: `${useability * 100}%`,
              backgroundColor: color,
            }}
          />
        </div>

        <label className="flex items-center gap-2 mt-3 cursor-pointer">
          <input
            type="checkbox"
            checked={isLocked}
            onChange={(e) => onLockedChange(e.target.checked)}
            className="w-4 h-4 rounded"
            style={{ accentColor: "#f59e0b" }}
          />
          <span
            className="flex items-center gap-1"
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.6rem",
              color: "var(--text-muted, #71717a)",
            }}
          >
            {isLocked ? <Lock size={12} /> : <Unlock size={12} />}
            Set Useability
          </span>
        </label>
        {isLocked && (
          <p
            className="mt-1 ml-6"
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.6rem",
              color: "var(--text-dim, #52525b)",
            }}
          >
            Drag the bar to set preset value
          </p>
        )}
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
  useabilityLocked?: boolean;
  lockedUseability?: number;
  onUseabilityLockChange?: (locked: boolean) => void;
  onLockedUseabilityChange?: (value: number) => void;
}

export function HoldFeaturesSidebar({
  mode,
  enabledFeatures,
  selectedHold,
  isDragging,
  dragParams,
  getColor,
  onDeleteHold,
  useabilityLocked = false,
  lockedUseability = 0.5,
  onUseabilityLockChange,
  onLockedUseabilityChange,
}: HoldFeaturesSidebarProps) {
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

  const showLockControls =
    mode === "add" &&
    enabledFeatures.useability &&
    onUseabilityLockChange &&
    onLockedUseabilityChange;

  if (!displayHold && !showLockControls) {
    return (
      <aside
        className="w-80 flex items-center justify-center"
        style={{
          background: "var(--surface, #111113)",
          borderLeft: "1px solid var(--border, rgba(255,255,255,0.08))",
        }}
      >
        <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
      </aside>
    );
  }

  const useability = hasUseability ? displayHold!.useability! : 0.5;
  const color = getColor(useabilityLocked ? lockedUseability : useability);

  return (
    <aside
      className="w-80 flex flex-col overflow-auto"
      style={{
        background: "var(--surface, #111113)",
        borderLeft: "1px solid var(--border, rgba(255,255,255,0.08))",
      }}
    >
      <div
        className="p-6"
        style={{ borderBottom: "1px solid var(--border, rgba(255,255,255,0.08))" }}
      >
        <div className="flex items-center justify-between mb-4">
          <h2
            className="uppercase tracking-wider"
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.6rem",
              letterSpacing: "0.15em",
              color: "var(--text-muted, #71717a)",
            }}
          >
            {mode === "add" ? "New Hold" : `Hold #${selectedHold?.hold_index}`}
          </h2>
          {mode === "select" && selectedHold && (
            <button
              onClick={onDeleteHold}
              className="p-2 rounded-md transition-colors"
              style={{ color: "#f87171", background: "transparent", border: "none", cursor: "pointer" }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(248,113,113,0.1)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "transparent")
              }
              title="Delete hold"
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>

        <div className="space-y-3">
          {displayHold && (
            <div>
              <label
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.6rem",
                  letterSpacing: "0.15em",
                  textTransform: "uppercase" as const,
                  color: "var(--text-muted, #71717a)",
                }}
              >
                Position
              </label>
              <div className="mt-1 grid grid-cols-2 gap-2">
                <div
                  className="rounded-lg p-3"
                  style={{ background: "var(--bg, #09090b)" }}
                >
                  <div
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "0.6rem",
                      color: "var(--text-muted, #71717a)",
                      marginBottom: "4px",
                    }}
                  >
                    X
                  </div>
                  <div
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "1.1rem",
                      color: "var(--text-primary, #f4f4f5)",
                    }}
                  >
                    {displayHold.x.toFixed(2)} ft
                  </div>
                </div>
                <div
                  className="rounded-lg p-3"
                  style={{ background: "var(--bg, #09090b)" }}
                >
                  <div
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "0.6rem",
                      color: "var(--text-muted, #71717a)",
                      marginBottom: "4px",
                    }}
                  >
                    Y
                  </div>
                  <div
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "1.1rem",
                      color: "var(--text-primary, #f4f4f5)",
                    }}
                  >
                    {displayHold.y.toFixed(2)} ft
                  </div>
                </div>
              </div>
            </div>
          )}

          {hasDirection && displayHold && (
            <PullDirectionArrow
              pullX={displayHold.pull_x!}
              pullY={displayHold.pull_y!}
              color={color}
              size={100}
            />
          )}

          {showLockControls && (
            <UseabilityBar
              useability={
                isDragging && hasUseability
                  ? displayHold!.useability!
                  : lockedUseability
              }
              color={getColor(
                isDragging && hasUseability && !useabilityLocked
                  ? displayHold!.useability!
                  : lockedUseability
              )}
              isLocked={useabilityLocked}
              onLockedChange={onUseabilityLockChange!}
              onUseabilityChange={onLockedUseabilityChange!}
            />
          )}

          {!showLockControls && hasUseability && (
            <div>
              <label
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.6rem",
                  letterSpacing: "0.15em",
                  textTransform: "uppercase" as const,
                  color: "var(--text-muted, #71717a)",
                }}
              >
                Useability
              </label>
              <div
                className="mt-2 rounded-lg p-3"
                style={{ background: "var(--bg, #09090b)" }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "1.5rem",
                      color: "var(--text-primary, #f4f4f5)",
                    }}
                  >
                    {(useability * 100).toFixed(0)}%
                  </span>
                </div>
                <div
                  className="h-2 rounded-full overflow-hidden"
                  style={{ background: "var(--border, rgba(255,255,255,0.08))" }}
                >
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

      <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
    </aside>
  );
}
