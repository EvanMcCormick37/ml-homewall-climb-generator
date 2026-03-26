import { useState, useCallback, useRef } from "react";
import { Eraser, Hand, Edit, Trash2, Lock, Unlock, Plus } from "lucide-react";
import type {
  HoldMode,
  HoldDetail,
  EnabledFeatures,
  FeatureLabel,
  Tag,
} from "@/types";

// ─── Pull Direction Arrow ──────────────────────────────────────────────────────

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
  const arrowLength = radius * 0.5;
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

// ─── Constants ────────────────────────────────────────────────────────────────

const TAG_META: { key: Tag; label: string; hotkey: string; desc: string }[] = [
  {
    key: "pinch",
    label: "PINCH",
    hotkey: "p",
    desc: "A bilateral pinch (can be held two ways). Align the pull vector perpendicular to the dominant edge of the pinch (If perfectly symetric, choose the upwards-facing edge. If perfectly vertical, choose the edge facing away from the center. If right in the center, choose the left side. /-;}-/).",
  },
  {
    key: "flat",
    label: "FLAT",
    hotkey: "f",
    desc: "A hold with a single, straight (flat) edge (Not with regards to being incut or slopey, but flat along the surface of the wall. i.e. a flat, horizontal crimp.",
  },
];

// ─── Enabled Features Menu ────────────────────────────────────────────────────

interface EnabledFeaturesMenuProps {
  enabledFeatures: EnabledFeatures;
  onToggle: (feature: FeatureLabel) => void;
  onClose: () => void;
}

const FEATURES: { key: FeatureLabel; label: string; desc: string }[] = [
  { key: "direction", label: "Pull Direction", desc: "The optimal pull vector for the hold" },
  { key: "useability", label: "Useability", desc: "How easy a hold is to use." },
  { key: "footholds", label: "Foot Holds", desc: "" },
  { key: "tags", label: "Tags", desc: "Specific tags for hold types with deviant features" },
];

export function EnabledFeaturesMenu({
  enabledFeatures,
  onToggle,
  onClose,
}: EnabledFeaturesMenuProps) {

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
    </div>
  );
}

// ─── Hotkeys Panel ────────────────────────────────────────────────────────────

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
              · {kbd("p")} {kbd("m")} {kbd("s")} {kbd("v")} {kbd("j")} Sticky
              tag toggle
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Draggable Useability Bar ─────────────────────────────────────────────────

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
      const percentage = Math.max(
        0,
        Math.min(1, (clientX - rect.left) / rect.width),
      );
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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div
        style={{
          background: "var(--bg)",
          padding: "10px 12px",
          borderRadius: "var(--radius)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            marginBottom: "8px",
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
          {isLocked && (
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: "4px",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.5rem",
                color: "#f59e0b",
              }}
            >
              <Lock size={10} /> MANUAL
            </span>
          )}
        </div>
        <div
          ref={barRef}
          style={{
            height: "4px",
            background: "rgba(255,255,255,0.08)",
            cursor: isLocked ? "ew-resize" : "default",
            borderRadius: "2px",
            overflow: "hidden",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
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
      </div>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
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
            color: isLocked ? "#f59e0b" : "var(--text-muted)",
          }}
        >
          {isLocked ? <Lock size={10} /> : <Unlock size={10} />}
          {isLocked ? "Drag bar to adjust" : "Set fixed useability"}
        </span>
      </label>
    </div>
  );
}

// ─── Hold Features Sidebar ────────────────────────────────────────────────────

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
  stickyTag: Tag | null;
  onStickyTagToggle: (tag: Tag) => void;
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
  activeHold,
  onTagToggle,
  stickyTag,
  onStickyTagToggle,
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

  const showEditUseability = mode === "edit" && isDragging && enabledFeatures.useability;

  const showTags = enabledFeatures.tags && activeHold != null;
  const showUseability =
    showLockControls || showEditUseability || (!showLockControls && hasUseability);
  const hasContent =
    displayHold || showUseability || showTags || enabledFeatures.tags;

  const sidebarStyle: React.CSSProperties = {
    width: "272px",
    flexShrink: 0,
    display: "flex",
    flexDirection: "column",
    overflow: "auto",
    background: "var(--surface)",
    borderLeft: "1px solid var(--border)",
  };

  const sectionStyle: React.CSSProperties = {
    padding: "14px 16px",
    borderBottom: "1px solid var(--border)",
  };

  const sectionHeaderStyle: React.CSSProperties = {
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.5rem",
    letterSpacing: "0.18em",
    textTransform: "uppercase",
    color: "var(--text-dim)",
    margin: "0 0 10px 0",
  };

  const useability = showEditUseability
    ? (dragParams.useability ?? 0.5)
    : hasUseability ? displayHold!.useability! : 0.5;
  const displayColor = getColor(
    useabilityLocked ? lockedUseability : useability,
  );

  if (!hasContent) {
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

  return (
    <aside style={sidebarStyle}>
      {/* ── Hold Info: position + direction ── */}
      {displayHold && (
        <div style={sectionStyle}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "10px",
            }}
          >
            <h2 style={sectionHeaderStyle}>
              {mode === "add"
                ? "New Hold"
                : `Hold #${selectedHold?.hold_index}`}
            </h2>
            {mode === "select" && selectedHold && (
              <button
                onClick={onDeleteHold}
                style={{
                  padding: "4px",
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
                <Trash2 size={14} />
              </button>
            )}
          </div>

          {/* Position grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "5px",
              marginBottom: hasDirection ? "10px" : 0,
            }}
          >
            {[
              { axis: "X", val: displayHold.x },
              { axis: "Y", val: displayHold.y },
            ].map(({ axis, val }) => (
              <div
                key={axis}
                style={{
                  background: "var(--bg)",
                  padding: "8px 10px",
                  borderRadius: "var(--radius)",
                }}
              >
                <div
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.5rem",
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

          {/* Pull direction arrow */}
          {hasDirection && displayHold && (
            <PullDirectionArrow
              pullX={displayHold.pull_x!}
              pullY={displayHold.pull_y!}
              color={displayColor}
              size={90}
            />
          )}
        </div>
      )}

      {/* ── Useability ── */}
      {showUseability && (
        <div style={sectionStyle}>
          <h2 style={sectionHeaderStyle}>Useability</h2>
          {showLockControls ? (
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
          ) : (
            <div
              style={{
                background: "var(--bg)",
                padding: "10px 12px",
                borderRadius: "var(--radius)",
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
                  height: "4px",
                  background: "rgba(255,255,255,0.08)",
                  overflow: "hidden",
                  marginTop: "8px",
                  borderRadius: "2px",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${useability * 100}%`,
                    backgroundColor: displayColor,
                    transition: "width 0.2s",
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Tags (active hold) ── */}
      {showTags && (
        <div style={sectionStyle}>
          <h2 style={sectionHeaderStyle}>
            Tags — Hold #{activeHold!.hold_index}
          </h2>

          {/* Tag buttons */}
          <div style={{ display: "flex", gap: "4px", marginBottom: "8px" }}>
            {TAG_META.map(({ key, label }) => {
              const isActive = activeHold!.tags.includes(key);
              const isSticky = stickyTag === key;
              return (
                <button
                  key={key}
                  onClick={() => onTagToggle?.(key)}
                  title={
                    isSticky ? `Sticky — auto-applies to new/edited holds` : ""
                  }
                  style={{
                    flex: 1,
                    padding: "5px 0",
                    background: isActive
                      ? "var(--cyan)"
                      : isSticky
                        ? "rgba(6,182,212,0.12)"
                        : "transparent",
                    color: isActive
                      ? "#09090b"
                      : isSticky
                        ? "var(--cyan)"
                        : "var(--text-muted)",
                    border: `1px solid ${isActive ? "var(--cyan)" : isSticky ? "rgba(6,182,212,0.35)" : "var(--border)"}`,
                    borderRadius: "var(--radius)",
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.5rem",
                    fontWeight: 700,
                    letterSpacing: "0.05em",
                    cursor: "pointer",
                    transition: "all 0.1s",
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>

          {/* Sticky indicator */}
          {stickyTag && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "5px 8px",
                background: "rgba(6,182,212,0.07)",
                border: "1px solid rgba(6,182,212,0.15)",
                borderRadius: "var(--radius)",
                marginBottom: "6px",
              }}
            >
              <span
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.5rem",
                  color: "var(--text-dim)",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                Sticky:
              </span>
              <button
                onClick={() => onStickyTagToggle(stickyTag)}
                title="Click to deactivate"
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.5rem",
                  fontWeight: 700,
                  color: "var(--cyan)",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  padding: 0,
                  textDecoration: "underline",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                }}
              >
                {stickyTag}
              </button>
            </div>
          )}

          {/* Hotkey legend */}
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {TAG_META.map(({ key, hotkey }) => {
              const isSticky = stickyTag === key;
              return (
                <span
                  key={key}
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.5rem",
                    color: isSticky ? "var(--cyan)" : "var(--text-dim)",
                  }}
                >
                  <span
                    style={{
                      color: isSticky ? "var(--cyan)" : "var(--text-muted)",
                      fontWeight: isSticky ? 700 : 400,
                    }}
                  >
                    {hotkey}
                  </span>
                  ={key}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Tag Guide ── */}
      {enabledFeatures.tags && (
        <div style={sectionStyle}>
          <h2 style={sectionHeaderStyle}>Tag Guide</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {TAG_META.map(({ key, label, hotkey, desc }) => {
              const isSticky = stickyTag === key;
              return (
                <div
                  key={key}
                  onClick={() => onStickyTagToggle(key)}
                  style={{
                    padding: "8px 10px",
                    background: isSticky ? "rgba(6,182,212,0.07)" : "var(--bg)",
                    borderRadius: "var(--radius)",
                    border: `1px solid ${isSticky ? "rgba(6,182,212,0.2)" : "transparent"}`,
                    cursor: "pointer",
                    transition: "background 0.15s, border-color 0.15s",
                  }}
                  onMouseEnter={(e) => {
                    if (!isSticky)
                      e.currentTarget.style.background = "rgba(6,182,212,0.04)";
                  }}
                  onMouseLeave={(e) => {
                    if (!isSticky)
                      e.currentTarget.style.background = "var(--bg)";
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                      marginBottom: "3px",
                    }}
                  >
                    <span
                      style={{
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "0.55rem",
                        fontWeight: 700,
                        color: isSticky ? "var(--cyan)" : "var(--text-primary)",
                      }}
                    >
                      {label}
                    </span>
                    <span
                      style={{
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "0.5rem",
                        color: "var(--text-dim)",
                        border: "1px solid var(--border)",
                        padding: "0 3px",
                        lineHeight: 1.4,
                      }}
                    >
                      {hotkey}
                    </span>
                  </div>
                  <div
                    style={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "0.5rem",
                      color: "var(--text-muted)",
                      lineHeight: 1.5,
                    }}
                  >
                    {desc}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <HotkeysAndInstructions enabledFeatures={enabledFeatures} />
    </aside>
  );
}
