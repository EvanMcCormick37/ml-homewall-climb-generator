import { useState, useCallback, useRef } from "react";
import {
  PlusCircle,
  Eraser,
  Hand,
  Edit,
  Trash2,
  Lock,
  Unlock,
  Plus,
} from "lucide-react";
import type {
  HoldMode,
  HoldDetail,
  EnabledFeatures,
  FeatureLabel,
  Tag,
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
    <svg
      width={size}
      height={size}
      style={{ display: "block", margin: "0 auto" }}
    >
      <circle
        cx={centerX}
        cy={centerY}
        r={radius}
        fill="none"
        stroke="rgba(255,255,255,0.1)"
        strokeWidth="1"
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

// --- Enabled Features Menu ---

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
  const FEATURES: { key: FeatureLabel; label: string; desc: string }[] = [
    {
      key: "direction",
      label: "Pull Direction",
      desc: "The optimal pull vector for the hold",
    },
    {
      key: "useability",
      label: "Useability",
      desc: "How easy a hold is to use.",
    },
    { key: "footholds", label: "Foot Holds", desc: "" },
    {
      key: "tags",
      label: "Tags",
      desc: "Specific tags for hold types with deviant features",
    },
  ];

  const TAG_EXPLANATIONS: { key: Tag; label: string; desc: string }[] = [
    {
      key: "pinch",
      label: "PINCH",
      desc: "A bilateral pinch (Can be held two ways). Align the pull vector to be parallel with the hold's edges if possible.",
    },
    {
      key: "macro",
      label: "MACRO",
      desc: "A very large hold. This is important as BetaZero treats holds as points, so very large holds can confuse it if not labelled as macros.",
    },
    {
      key: "sloper",
      label: "SLOPER",
      desc: "A hold that is rather, well, slopey. Wall angle affects slopers more severely than other hold types.",
    },
    {
      key: "versatile",
      label: "VERSATILE",
      desc: "A hold which can be taken in multiple directions, but isn't quite as regular as a bilateral pinch. Think of those moonboard holds which can be taken in many different ways.",
    },
    {
      key: "jug",
      label: "JUG",
      desc: "A true jug. Not any good hold. A hold which has a severe incut and can thus be used in a multitude of pull directions (and for dynos).",
    },
  ];

  return (
    <div
      style={{
        position: "absolute",
        top: "60px",
        left: "16px",
        borderRadius: "var(--radius)",
        boxShadow: "0 20px 40px rgba(0,0,0,0.6)",
        zIndex: 50,
        width: "248px",
        background: "var(--surface)",
        border: "1px solid var(--border)",
      }}
    >
      <div
        style={{
          padding: "10px 14px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <h3
          style={{
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.55rem",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
            margin: 0,
          }}
        >
          HOLD FEATURES TO ENABLE
        </h3>
        <button
          onClick={onClose}
          style={{
            background: "transparent",
            border: "none",
            color: "var(--text-dim)",
            cursor: "pointer",
            fontSize: "0.7rem",
            lineHeight: 1,
          }}
        >
          ✕
        </button>
      </div>
      <div
        style={{
          padding: "10px 14px",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}
      >
        {FEATURES.map(({ key, label, desc }) => (
          <label
            key={key}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={enabledFeatures[key]}
              onChange={() => onToggle(key)}
              style={{
                width: "14px",
                height: "14px",
                accentColor: "var(--cyan)",
                flexShrink: 0,
              }}
            />
            <div>
              <div
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.65rem",
                  fontWeight: 700,
                  color: "var(--text-primary)",
                }}
              >
                {label}
              </div>
              <div
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.55rem",
                  color: "var(--text-muted)",
                }}
              >
                {desc}
              </div>
            </div>
          </label>
        ))}
      </div>
      {enabledFeatures["tags"] && (
        <div
          style={{
            padding: "8px 14px",
            background: "var(--bg)",
            borderTop: "1px solid var(--border)",
          }}
          className="bz-mono"
        >
          Tag Descriptions{" "}
          <div
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.6rem",
              color: "var(--text-muted)",
            }}
          >
            These are optional tags designed to handle holds which normally
            confuse the ClimbDDPM model.
          </div>
          {TAG_EXPLANATIONS.map(({ key, label, desc }) => (
            <label
              key={key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                cursor: "pointer",
              }}
            >
              <div>
                <div
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.65rem",
                    fontWeight: 700,
                    color: "var(--text-primary)",
                  }}
                >
                  {label}
                </div>
                <div
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.55rem",
                    color: "var(--text-muted)",
                  }}
                >
                  {desc}
                </div>
              </div>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Hotkeys Panel ---

type HotkeysAndInstructionsProps = { enabledFeatures: EnabledFeatures };

function HotkeysAndInstructions({
  enabledFeatures,
}: HotkeysAndInstructionsProps) {
  const kbd = (label: string, color?: string) => (
    <span
      style={{
        fontFamily: "'Space Mono', monospace",
        fontWeight: 700,
        fontSize: "0.6rem",
        color: color ?? "var(--text-primary)",
        border: `1px solid ${color ? color.replace(")", ",0.3)").replace("rgb", "rgba") : "var(--border)"}`,
        padding: "1px 5px",
        lineHeight: 1.4,
      }}
    >
      {label}
    </span>
  );

  return (
    <div
      style={{
        margin: "12px",
        padding: "12px 14px",
        background: "var(--bg)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
      }}
    >
      <h3
        style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: "0.55rem",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--text-dim)",
          margin: "0 0 8px 0",
        }}
      >
        Hotkeys
      </h3>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          fontSize: "11px",
        }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {kbd("1", "#34d399")}
          <Plus size={12} style={{ color: "#34d399", flexShrink: 0 }} />
          <span style={{ color: "var(--text-muted)" }}>
            {`Click${enabledFeatures.direction || enabledFeatures.useability ? " & drag" : ""} — new hold`}
          </span>
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {kbd("2", "#f59e0b")}
          <Edit size={12} style={{ color: "#f59e0b", flexShrink: 0 }} />
          <span style={{ color: "var(--text-muted)" }}>
            {`Click${enabledFeatures.direction || enabledFeatures.useability ? " & drag" : ""} — edit hold`}
          </span>
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {kbd("3", "rgb(248,113,113)")}
          <Eraser size={12} style={{ color: "#f87171", flexShrink: 0 }} />
          <span style={{ color: "var(--text-muted)" }}>
            Click — delete hold
          </span>
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {kbd("4", "rgb(96,165,250)")}
          <Hand size={12} style={{ color: "#60a5fa", flexShrink: 0 }} />
          <span style={{ color: "var(--text-muted)" }}>
            Click — select / view hold
          </span>
        </span>
        <div
          style={{
            marginTop: "4px",
            paddingTop: "6px",
            borderTop: "1px solid var(--border)",
            display: "flex",
            flexDirection: "column",
            gap: "3px",
          }}
        >
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              color: "var(--text-muted)",
            }}
          >
            · {kbd("Shift + Drag")} Pan
          </span>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              color: "var(--text-muted)",
            }}
          >
            · {kbd("Scroll")} Zoom
          </span>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              color: "var(--text-muted)",
            }}
          >
            · {kbd("Ctrl+Z")} Delete last added hold
          </span>
          {enabledFeatures.footholds && (
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                color: "var(--text-muted)",
              }}
            >
              · {kbd("x")} Toggle hand / foot
            </span>
          )}
          {enabledFeatures.tags && (
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                color: "var(--text-muted)",
              }}
            >
              · {kbd("p")} {kbd("m")} {kbd("s")} {kbd("v")}
              {kbd("j")} Tag hotkeys (Pinch, Macro, Sloper, Versatile, Jug)
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Draggable Useability Bar ---

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
    [isLocked, onUseabilityChange],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!isLocked) return;
      setIsDragging(true);
      handleBarInteraction(e.clientX);
    },
    [isLocked, handleBarInteraction],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging || !isLocked) return;
      handleBarInteraction(e.clientX);
    },
    [isDragging, isLocked, handleBarInteraction],
  );

  const handleMouseUp = useCallback(() => setIsDragging(false), []);
  const handleMouseLeave = useCallback(() => setIsDragging(false), []);

  return (
    <div>
      <label
        style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: "0.6rem",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
        }}
      >
        Useability
      </label>
      <div
        style={{
          marginTop: "6px",
          background: "var(--bg)",
          padding: "10px 12px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "8px",
          }}
        >
          <span
            style={{
              fontSize: "1.4rem",
              fontWeight: 700,
              color: "var(--text-primary)",
              fontFamily: "'Space Mono', monospace",
            }}
          >
            {(useability * 100).toFixed(0)}%
          </span>
          {isLocked && (
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: "4px",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.55rem",
                color: "#f59e0b",
              }}
            >
              <Lock size={11} /> Manual
            </span>
          )}
        </div>
        <div
          ref={barRef}
          style={{
            height: "5px",
            background: "var(--border)",
            cursor: isLocked ? "ew-resize" : "default",
            overflow: "hidden",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          <div
            style={{
              height: "100%",
              width: `${useability * 100}%`,
              backgroundColor: color,
              transition: "width 0.075s",
            }}
          />
        </div>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            marginTop: "8px",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={isLocked}
            onChange={(e) => onLockedChange(e.target.checked)}
            style={{ width: "13px", height: "13px", accentColor: "#f59e0b" }}
          />
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "4px",
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.55rem",
              color: "var(--text-muted)",
            }}
          >
            {isLocked ? <Lock size={11} /> : <Unlock size={11} />}
            Set Useability
          </span>
        </label>
        {isLocked && (
          <p
            style={{
              marginTop: "3px",
              marginLeft: "21px",
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.55rem",
              color: "var(--text-dim)",
            }}
          >
            Drag bar to adjust preset
          </p>
        )}
      </div>
    </div>
  );
}

// --- Hold Features Sidebar ---

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
  activeHold?: HoldDetail | null;
  onTagToggle?: (tag: Tag) => void;
}

const TAGS: { value: Tag; label: string }[] = [
  { value: "pinch", label: "PINCH" },
  { value: "macro", label: "MACRO" },
  { value: "sloper", label: "SLOPER" },
  { value: "versatile", label: "VERSATILE" },
  { value: "jug", label: "JUG" },
];

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
  activeHold,
  onTagToggle,
}: HoldFeaturesSidebarProps) {
  const displayHold = mode === "add" && isDragging ? dragParams : selectedHold;
  const hasDirection =
    displayHold &&
    "pull_x" in displayHold &&
    displayHold.pull_x != null &&
    "pull_y" in displayHold &&
    displayHold.pull_y != null;
  const hasUseability =
    displayHold &&
    "useability" in displayHold &&
    displayHold.useability != null;

  const showLockControls =
    mode === "add" &&
    enabledFeatures.useability &&
    onUseabilityLockChange &&
    onLockedUseabilityChange;

  const showTags = enabledFeatures.tags && activeHold != null;

  const sidebarStyle: React.CSSProperties = {
    width: "272px",
    flexShrink: 0,
    display: "flex",
    flexDirection: "column",
    overflow: "auto",
    background: "var(--surface)",
    borderLeft: "1px solid var(--border)",
  };

  if (!displayHold && !showLockControls && !showTags) {
    return (
      <aside
        style={{
          ...sidebarStyle,
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
      </aside>
    );
  }

  const useability = hasUseability ? displayHold!.useability! : 0.5;
  const color = getColor(useabilityLocked ? lockedUseability : useability);

  return (
    <aside style={sidebarStyle}>
      {displayHold ? (
        <div
          style={{
            padding: "18px 20px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "12px",
            }}
          >
            <h2
              style={{
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.55rem",
                letterSpacing: "0.15em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
                margin: 0,
              }}
            >
              {mode === "add"
                ? "New Hold"
                : `Hold #${selectedHold?.hold_index}`}
            </h2>
            {mode === "select" && selectedHold && (
              <button
                onClick={onDeleteHold}
                style={{
                  padding: "5px",
                  color: "#f87171",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  borderRadius: "var(--radius)",
                  transition: "background 0.15s",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "rgba(248,113,113,0.1)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "transparent")
                }
                title="Delete hold"
              >
                <Trash2 size={15} />
              </button>
            )}
          </div>

          <div
            style={{ display: "flex", flexDirection: "column", gap: "10px" }}
          >
            {/* Position */}
            <div>
              <label
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.55rem",
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "var(--text-muted)",
                }}
              >
                Position
              </label>
              <div
                style={{
                  marginTop: "5px",
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "5px",
                }}
              >
                {[
                  { axis: "X", val: displayHold.x },
                  { axis: "Y", val: displayHold.y },
                ].map(({ axis, val }) => (
                  <div
                    key={axis}
                    style={{ background: "var(--bg)", padding: "8px 10px" }}
                  >
                    <div
                      style={{
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "0.55rem",
                        color: "var(--text-dim)",
                        marginBottom: "3px",
                      }}
                    >
                      {axis}
                    </div>
                    <div
                      style={{
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "1rem",
                        fontWeight: 700,
                        color: "var(--text-primary)",
                      }}
                    >
                      {val.toFixed(2)} ft
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Direction */}
            {hasDirection && displayHold && (
              <PullDirectionArrow
                pullX={displayHold.pull_x!}
                pullY={displayHold.pull_y!}
                color={color}
                size={90}
              />
            )}

            {/* Useability — locked (add mode) */}
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
                    : lockedUseability,
                )}
                isLocked={useabilityLocked}
                onLockedChange={onUseabilityLockChange!}
                onUseabilityChange={onLockedUseabilityChange!}
              />
            )}

            {/* Useability — read-only (select/edit mode) */}
            {!showLockControls && hasUseability && (
              <div>
                <label
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.55rem",
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "var(--text-muted)",
                  }}
                >
                  Useability
                </label>
                <div
                  style={{
                    marginTop: "6px",
                    background: "var(--bg)",
                    padding: "10px 12px",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "1.4rem",
                      fontWeight: 700,
                      color: "var(--text-primary)",
                    }}
                  >
                    {(useability * 100).toFixed(0)}%
                  </span>
                  <div
                    style={{
                      height: "5px",
                      background: "var(--border)",
                      overflow: "hidden",
                      marginTop: "8px",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${useability * 100}%`,
                        backgroundColor: color,
                        transition: "width 0.2s",
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div>
          {" "}
          <div>
            {" "}
            {/* Useability — locked (add mode) */}
            {showLockControls && (
              <UseabilityBar
                useability={lockedUseability}
                color={getColor(lockedUseability)}
                isLocked={useabilityLocked}
                onLockedChange={onUseabilityLockChange!}
                onUseabilityChange={onLockedUseabilityChange!}
              />
            )}
            {/* Useability — read-only (select/edit mode) */}
            {!showLockControls && hasUseability && (
              <div>
                <label
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.55rem",
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "var(--text-muted)",
                  }}
                >
                  Useability
                </label>
                <div
                  style={{
                    marginTop: "6px",
                    background: "var(--bg)",
                    padding: "10px 12px",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "1.4rem",
                      fontWeight: 700,
                      color: "var(--text-primary)",
                    }}
                  >
                    {(useability * 100).toFixed(0)}%
                  </span>
                  <div
                    style={{
                      height: "5px",
                      background: "var(--border)",
                      overflow: "hidden",
                      marginTop: "8px",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${useability * 100}%`,
                        backgroundColor: color,
                        transition: "width 0.2s",
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tags */}
      {showTags && (
        <div
          style={{
            padding: "14px 20px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <label
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.55rem",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              display: "block",
              marginBottom: "8px",
            }}
          >
            Tags — Hold #{activeHold!.hold_index}
          </label>
          <div style={{ display: "flex", gap: "5px" }}>
            {TAGS.map(({ value, label }) => {
              const isActive = activeHold!.tags.includes(value);
              return (
                <button
                  key={value}
                  onClick={() => onTagToggle?.(value)}
                  style={{
                    flex: 1,
                    padding: "6px 0",
                    background: isActive ? "var(--cyan)" : "transparent",
                    color: isActive ? "#09090b" : "var(--text-muted)",
                    border: `1px solid ${isActive ? "var(--cyan)" : "var(--border)"}`,
                    borderRadius: "var(--radius)",
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.55rem",
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    cursor: "pointer",
                    transition: "all 0.1s",
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: "10px", marginTop: "5px" }}>
            {[
              ["p", "pinch"],
              ["m", "macro"],
              ["s", "sloper"],
              ["v", "versatile"],
              ["j", "jug"],
            ].map(([key, name]) => (
              <span
                key={key}
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.5rem",
                  color: "var(--text-dim)",
                }}
              >
                <span style={{ color: "var(--text-muted)" }}>{key}</span>={name}
              </span>
            ))}
          </div>
        </div>
      )}

      <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
    </aside>
  );
}
