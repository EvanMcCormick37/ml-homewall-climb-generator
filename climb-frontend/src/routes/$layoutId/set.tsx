import {
  createFileRoute,
  useNavigate,
  type UseNavigateResult,
} from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useLayout } from "@/hooks";
import { generateClimbs } from "@/api/generate";
import { createClimb } from "@/api/climbs";
import { WakingScreen } from "@/components";
import { useUser } from "@clerk/clerk-react";
import {
  ArrowLeft,
  Sparkles,
  Loader2,
  Pencil,
  RotateCcw,
  X,
  RefreshCcw,
  Cpu,
  ChevronDown,
  Zap,
  Target,
  Turtle,
  SunMedium,
  ChevronLeft,
  ChevronRight,
  Plus,
  Layers,
} from "lucide-react";
import type {
  Holdset,
  GenerateRequest,
  GenerateSettings,
  GradeScale,
  LayoutDetail,
  SizeMetadata,
} from "@/types";
import {
  DEFAULT_GENERATE_SETTINGS,
  FAST_GENERATE_SETTINGS,
  SLOW_GENERATE_SETTINGS,
} from "@/types";
import { GLOBAL_STYLES } from "@/styles";
import {
  // Types & constants
  CATEGORY_ORDER,
  CATEGORY_LABELS,
  DEFAULT_DISPLAY_SETTINGS,
  VGRADE_OPTIONS,
  FONT_OPTIONS,
  generateClimbName,
  type HoldCategory,
  type DisplaySettings,
  type NamedHoldset,
  // UI primitives
  SectionLabel,
  TogglePair,
  BzRange,
  // Components
  MobileSwipeNav,
  WallCanvas,
  DisplaySettingsPanel,
  SaveShareMenu,
  DesktopSaveSharePanel,
  type SaveShareMenuProps,
  // Sharing
  decodeClimbFromParam,
  buildShareUrl,
  renderExportImage,
} from "@/components/wall";

// ─── Route ───────────────────────────────────────────────────────────────────

interface ClimbSearchParams {
  climb?: string;
}

export const Route = createFileRoute("/$layoutId/set")({
  component: SetPage,
  validateSearch: (search: Record<string, unknown>): ClimbSearchParams => ({
    climb: typeof search.climb === "string" ? search.climb : undefined,
  }),
  staleTime: 3_600_000,
});

// ─── Model Settings Panel ────────────────────────────────────────────────────

function ModelSettingsPanel({
  settings,
  onChange,
}: {
  settings: GenerateSettings;
  onChange: (s: GenerateSettings) => void;
}) {
  const update = (patch: Partial<GenerateSettings>) =>
    onChange({ ...settings, ...patch });
  const isDefault =
    settings.timesteps === DEFAULT_GENERATE_SETTINGS.timesteps &&
    settings.guidance_value === DEFAULT_GENERATE_SETTINGS.guidance_value &&
    settings.x_offset === DEFAULT_GENERATE_SETTINGS.x_offset &&
    settings.t_start_projection ===
      DEFAULT_GENERATE_SETTINGS.t_start_projection &&
    settings.deterministic === DEFAULT_GENERATE_SETTINGS.deterministic &&
    settings.seed === DEFAULT_GENERATE_SETTINGS.seed;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <BzRange
        label="Generation Timesteps"
        desc="Fewer timesteps = faster generation but reduced quality. WARNING: Using this setting to generate high difficulty climbs may result in errors due to lack of holds. I know that sounds like a fake error but if you try it you'll see what I mean."
        value={settings.timesteps}
        min={25}
        max={200}
        step={5}
        onChange={(v) => update({ timesteps: v })}
        displayValue={String(settings.timesteps)}
        leftLabel="Faster"
        rightLabel="Higher Quality"
      />
      <BzRange
        label="Guidance Value"
        desc="Guidance value for CFG. A higher guidance value will exaggerate the climb features corresponding to the wall angle and grade. However, a high Guidance value may create extreme or impossible hold arrangements."
        value={settings.guidance_value}
        min={1.0}
        max={10.0}
        step={0.5}
        onChange={(v) => update({ guidance_value: v })}
        displayValue={settings.guidance_value.toFixed(1)}
        leftLabel="Understated"
        rightLabel="Exaggerated"
      />

      <BzRange
        label="Projection Start Time"
        desc="Fraction of the diffusion process at which layout-projection begins. Higher values begin projecting onto the wall's holds earlier in the generative process."
        value={settings.t_start_projection}
        min={0.0}
        max={0.8}
        step={0.05}
        onChange={(v) => update({ t_start_projection: v })}
        displayValue={settings.t_start_projection.toFixed(2)}
        leftLabel="Later"
        rightLabel="Earlier"
      />

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel desc="Pin the generated climb to a specific horizontal position on the wall. 'Auto' chooses the optimal X-offset automatically.">
            X Offset
          </SectionLabel>
        </div>
        <TogglePair
          options={[
            { value: "auto", label: "Auto" },
            { value: "manual", label: "Manual" },
          ]}
          value={settings.x_offset == null ? "auto" : "manual"}
          onChange={(v) => update({ x_offset: v === "auto" ? null : 0.0 })}
        />
        {settings.x_offset != null && (
          <div style={{ marginTop: "12px" }}>
            <BzRange
              label=""
              desc=""
              value={settings.x_offset}
              min={-1.5}
              max={1.5}
              step={0.1}
              onChange={(v) => update({ x_offset: v })}
              displayValue={settings.x_offset.toFixed(1)}
              leftLabel="Left"
              rightLabel="Right"
            />
          </div>
        )}
      </div>

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel desc="Whether to reuse the initial noise vector at each diffusion step. Produces the same climb for a given Generate Settings configuration.">
            Generation Style
          </SectionLabel>
        </div>
        <TogglePair
          options={[
            { value: "det", label: "Deterministic" },
            { value: "non", label: "Nondeterministic" },
          ]}
          value={settings.deterministic ? "det" : "non"}
          onChange={(v) => update({ deterministic: v === "det" })}
        />
        {settings.deterministic && (
          <div style={{ marginTop: "12px" }}>
            <SectionLabel desc="Seed value used for the initial noise vector.">
              Seed
            </SectionLabel>
            <input
              type="number"
              value={settings.seed}
              onChange={(e) =>
                update({ seed: parseInt(e.target.value, 10) || 0 })
              }
              style={{
                marginTop: "8px",
                width: "100%",
                background: "var(--surface)",
                color: "var(--text-primary)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "7px 10px",
                fontSize: "0.75rem",
                fontFamily: "'Space Mono', monospace",
                outline: "none",
              }}
            />
          </div>
        )}
      </div>

      {!isDefault && (
        <button
          onClick={() => onChange({ ...DEFAULT_GENERATE_SETTINGS })}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "6px",
            padding: "7px",
            width: "100%",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            border: "1px solid var(--border)",
            background: "transparent",
            color: "var(--text-muted)",
            cursor: "pointer",
            borderRadius: "var(--radius)",
            transition: "all 0.15s",
          }}
        >
          <RotateCcw size={10} /> Reset Defaults
        </button>
      )}
    </div>
  );
}

// ─── GenerationPanel ─────────────────────────────────────────────────────────

interface GenerationPanelProps {
  displaySettings: DisplaySettings;
  gradingScale: GradeScale;
  gradeOptions: string[];
  grade: string;
  onGradingScaleChange: (s: GradeScale) => void;
  onGradeChange: (g: string) => void;
  numClimbs: number | null;
  onNumClimbsChange: (n: number | null) => void;
  angle: number | null;
  angleFixed: boolean;
  onAngleChange: (a: number | null) => void;
  generateSettings: GenerateSettings;
  onGenerateSettingsChange: (s: GenerateSettings) => void;
  showModelSettings: boolean;
  onToggleModelSettings: () => void;
  isGenerating: boolean;
  error: string | null;
  onGenerate: () => void;
  onCreateBlank: () => void;
  holdsets: NamedHoldset[];
  selectedIndex: number | null;
  onSelectHoldset: (i: number) => void;
  onDeleteHoldset: (i: number) => void;
  onClearHoldsets: () => void;
}

function GenerationPanel({
  displaySettings,
  gradingScale,
  gradeOptions,
  grade,
  onGradingScaleChange,
  onGradeChange,
  numClimbs,
  onNumClimbsChange,
  angle,
  angleFixed,
  onAngleChange,
  generateSettings,
  onGenerateSettingsChange,
  showModelSettings,
  onToggleModelSettings,
  isGenerating,
  error,
  onGenerate,
  onCreateBlank,
  holdsets,
  selectedIndex,
  onSelectHoldset,
  onDeleteHoldset,
  onClearHoldsets,
}: GenerationPanelProps) {
  const [showClimbParams, setShowClimbParams] = useState(true);
  const panelRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef({ isDragging: false, startY: 0, startScrollTop: 0 });

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest("button,input,select,a")) return;
    const panel = panelRef.current;
    if (!panel) return;
    dragRef.current = {
      isDragging: true,
      startY: e.clientY,
      startScrollTop: panel.scrollTop,
    };
    panel.style.cursor = "grabbing";
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current.isDragging) return;
    const panel = panelRef.current;
    if (!panel) return;
    panel.scrollTop =
      dragRef.current.startScrollTop - (e.clientY - dragRef.current.startY);
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current.isDragging = false;
    if (panelRef.current) panelRef.current.style.cursor = "";
  }, []);

  const isPresetActive = (preset: GenerateSettings) =>
    generateSettings.timesteps === preset.timesteps &&
    generateSettings.guidance_value === preset.guidance_value &&
    generateSettings.x_offset === preset.x_offset &&
    generateSettings.t_start_projection === preset.t_start_projection &&
    generateSettings.deterministic === preset.deterministic &&
    generateSettings.seed === preset.seed;
  const isCustom =
    !isPresetActive(FAST_GENERATE_SETTINGS) &&
    !isPresetActive(DEFAULT_GENERATE_SETTINGS) &&
    !isPresetActive(SLOW_GENERATE_SETTINGS);

  const presetBtn = (active: boolean) => ({
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    gap: "6px",
    padding: "10px 4px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.55rem",
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
    border: `1px solid ${active ? "var(--cyan)" : "var(--border)"}`,
    background: active ? "var(--cyan-dim)" : "transparent",
    color: active ? "var(--cyan)" : "var(--text-muted)",
    cursor: "pointer",
    borderRadius: "var(--radius)",
    transition: "all 0.15s",
  });

  const inputStyle = {
    width: "100%",
    background: "var(--surface)",
    color: "var(--text-primary)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "7px 10px",
    fontSize: "0.75rem",
    fontFamily: "'Space Mono', monospace",
    outline: "none",
    cursor: "pointer",
  };

  return (
    <div
      ref={panelRef}
      style={{ flex: 1, minHeight: 0, overflowY: "auto", userSelect: "none" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Climb Parameters */}
      <div style={{ borderBottom: "1px solid var(--border)" }}>
        <button
          onClick={() => setShowClimbParams((v) => !v)}
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "12px 16px",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            color: "var(--text-muted)",
          }}
        >
          <span
            className="bz-oswald"
            style={{ fontSize: "0.65rem", letterSpacing: "0.15em" }}
          >
            Climb Parameters
          </span>
          <ChevronDown
            size={14}
            style={{
              color: "var(--text-muted)",
              transform: showClimbParams ? "rotate(180deg)" : "none",
              transition: "transform 0.2s",
            }}
          />
        </button>

        {showClimbParams && (
          <div
            style={{
              padding: "0 16px 16px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            }}
          >
            {/* Grading scale */}
            <div>
              <div style={{ marginBottom: "8px" }}>
                <SectionLabel>Grading Scale</SectionLabel>
              </div>
              <TogglePair
                options={[
                  { value: "v_grade", label: "V-grade" },
                  { value: "font", label: "Fontainebleau" },
                ]}
                value={gradingScale}
                onChange={(v) => onGradingScaleChange(v as GradeScale)}
              />
            </div>

            {/* Grade */}
            <div>
              <div style={{ marginBottom: "6px" }}>
                <SectionLabel>Target Grade</SectionLabel>
              </div>
              <select
                value={grade}
                onChange={(e) => onGradeChange(e.target.value)}
                style={inputStyle}
              >
                {gradeOptions.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </div>

            {/* Num climbs */}
            <div>
              <div style={{ marginBottom: "6px" }}>
                <SectionLabel>Number of Climbs</SectionLabel>
              </div>
              <input
                type="number"
                min={1}
                max={10}
                style={inputStyle}
                value={numClimbs ?? ""}
                onChange={(e) =>
                  onNumClimbsChange(
                    e.target.value === ""
                      ? null
                      : Math.max(1, Math.min(10, parseInt(e.target.value))),
                  )
                }
              />
            </div>

            {/* layout angle */}
            <div>
              <div style={{ marginBottom: "6px" }}>
                <SectionLabel>layout Angle (°)</SectionLabel>
              </div>
              <input
                type="number"
                min={0}
                max={90}
                disabled={angleFixed}
                style={{ ...inputStyle, opacity: angleFixed ? 0.4 : 1 }}
                value={angle ?? ""}
                onChange={(e) => {
                  if (e.target.value === "") {
                    onAngleChange(null);
                    return;
                  }
                  const p = parseInt(e.target.value);
                  if (!isNaN(p)) onAngleChange(Math.max(0, Math.min(90, p)));
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Generation Mode */}
      <div
        style={{
          padding: "16px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          gap: "14px",
        }}
      >
        <span
          className="bz-oswald text-zinc-400"
          style={{ fontSize: "0.65rem", letterSpacing: "0.15em" }}
        >
          Generation Mode
        </span>

        {/* Presets */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "6px",
          }}
        >
          {[
            {
              preset: FAST_GENERATE_SETTINGS,
              icon: <Zap size={14} />,
              label: "Speed",
            },
            {
              preset: DEFAULT_GENERATE_SETTINGS,
              icon: <Target size={14} />,
              label: "Standard",
            },
            {
              preset: SLOW_GENERATE_SETTINGS,
              icon: <Turtle size={14} />,
              label: "Quality",
            },
          ].map(({ preset, icon, label }) => (
            <button
              key={label}
              onClick={() => onGenerateSettingsChange(preset)}
              style={presetBtn(isPresetActive(preset))}
            >
              {icon}
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Custom settings toggle */}
        <button
          onClick={onToggleModelSettings}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "7px 10px",
            width: "100%",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            border: `1px solid ${isCustom || showModelSettings ? "var(--border-active)" : "var(--border)"}`,
            background:
              isCustom || showModelSettings ? "var(--cyan-dim)" : "transparent",
            color:
              isCustom || showModelSettings
                ? "var(--cyan)"
                : "var(--text-muted)",
            cursor: "pointer",
            borderRadius: "var(--radius)",
            transition: "all 0.15s",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <Cpu size={11} />
            {"Custom Generation Settings"}
          </span>
          <ChevronDown
            size={11}
            style={{
              transform: showModelSettings ? "rotate(180deg)" : "none",
              transition: "transform 0.2s",
            }}
          />
        </button>

        {showModelSettings && (
          <div
            style={{
              padding: "14px",
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
            }}
          >
            <ModelSettingsPanel
              settings={generateSettings}
              onChange={onGenerateSettingsChange}
            />
          </div>
        )}

        {/* Generate button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
            padding: "10px 16px",
            width: "100%",
            fontFamily: "'Oswald', sans-serif",
            fontSize: "0.85rem",
            fontWeight: 700,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            background: isGenerating ? "var(--surface2)" : "var(--cyan)",
            color: isGenerating ? "var(--text-muted)" : "#09090b",
            border: "none",
            borderRadius: "var(--radius)",
            cursor: isGenerating ? "not-allowed" : "pointer",
            transition: "all 0.15s",
          }}
        >
          {isGenerating ? (
            <>
              <Loader2
                size={14}
                style={{ animation: "spin 1s linear infinite" }}
              />{" "}
              Generating…
            </>
          ) : (
            <>
              <Sparkles size={14} /> Generate
            </>
          )}
        </button>

        {/* Create Blank button */}
        <button
          onClick={onCreateBlank}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
            padding: "9px 16px",
            width: "100%",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.65rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            background: "transparent",
            color: "var(--text-muted)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          <Plus size={12} /> Create Blank
        </button>
        {error && (
          <div
            className="bz-mono"
            style={{
              fontSize: "0.65rem",
              color: "#f87171",
              background: "rgba(248,113,113,0.08)",
              border: "1px solid rgba(248,113,113,0.2)",
              borderRadius: "var(--radius)",
              padding: "8px 10px",
            }}
          >
            {error}
          </div>
        )}
      </div>

      {/* Holdset list */}
      <HoldsetList
        holdsets={holdsets}
        displaySettings={displaySettings}
        selectedIndex={selectedIndex}
        onSelect={onSelectHoldset}
        onDelete={onDeleteHoldset}
        onClear={onClearHoldsets}
      />
    </div>
  );
}

// ─── HoldsetList ─────────────────────────────────────────────────────────────

function HoldsetList({
  holdsets,
  displaySettings,
  selectedIndex,
  onSelect,
  onDelete,
  onClear,
}: {
  holdsets: NamedHoldset[];
  displaySettings: DisplaySettings;
  selectedIndex: number | null;
  onSelect: (i: number) => void;
  onDelete: (i: number) => void;
  onClear: () => void;
}) {
  if (holdsets.length === 0) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 16px",
          gap: "10px",
        }}
      >
        <Sparkles size={28} style={{ color: "var(--text-dim)" }} />
        <p
          className="bz-mono"
          style={{
            fontSize: "0.65rem",
            color: "var(--text-muted)",
            textAlign: "center",
          }}
        >
          No climbs yet.
          <br />
          Configure and hit Generate.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 16px",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <SectionLabel>
          {holdsets.length} Climb{holdsets.length !== 1 ? "s" : ""}
        </SectionLabel>
        <button
          onClick={onClear}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "4px",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.55rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            background: "transparent",
            border: "none",
            color: "var(--text-dim)",
            cursor: "pointer",
            transition: "color 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "#f87171")}
          onMouseLeave={(e) =>
            (e.currentTarget.style.color = "var(--text-dim)")
          }
        >
          <RefreshCcw size={9} /> Clear
        </button>
      </div>

      {/* List */}
      {holdsets.map((entry, i) => {
        const isSelected = selectedIndex === i;
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "stretch",
              borderBottom: "1px solid var(--border)",
              borderLeft: `2px solid ${isSelected ? "var(--cyan)" : "transparent"}`,
              background: isSelected ? "var(--cyan-dim)" : "transparent",
              transition: "all 0.15s",
            }}
          >
            <button
              onClick={() => onSelect(i)}
              style={{
                flex: 1,
                textAlign: "left",
                padding: "12px 14px",
                display: "flex",
                alignItems: "center",
                gap: "10px",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                minWidth: 0,
              }}
            >
              <div
                style={{
                  width: "30px",
                  height: "30px",
                  flexShrink: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: isSelected
                    ? "var(--cyan-dim)"
                    : "var(--surface2)",
                  border: `1px solid ${isSelected ? "var(--cyan)" : "var(--border)"}`,
                  borderRadius: "var(--radius)",
                }}
              >
                <span
                  className="bz-mono"
                  style={{
                    fontSize: "0.6rem",
                    color: isSelected ? "var(--cyan)" : "var(--text-muted)",
                  }}
                >
                  {i + 1}
                </span>
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  className="bz-oswald"
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-primary)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {entry.name}
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginTop: "3px",
                  }}
                >
                  <span
                    className="bz-mono"
                    style={{ fontSize: "0.6rem", color: "var(--cyan)" }}
                  >
                    {entry.grade} @ {entry.angle}°
                  </span>
                  {(["start", "hand", "foot", "finish"] as HoldCategory[]).map(
                    (cat) => (
                      <span
                        key={cat}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "3px",
                        }}
                      >
                        <span
                          style={{
                            width: "6px",
                            height: "6px",
                            borderRadius: "50%",
                            background: displaySettings.categoryColors[cat],
                            flexShrink: 0,
                          }}
                        />
                        <span
                          className="bz-mono"
                          style={{
                            fontSize: "0.55rem",
                            color: "var(--text-muted)",
                          }}
                        >
                          {
                            entry.holdset[
                              cat === "start"
                                ? "start"
                                : cat === "finish"
                                  ? "finish"
                                  : cat === "hand"
                                    ? "hand"
                                    : "foot"
                            ].length
                          }
                        </span>
                      </span>
                    ),
                  )}
                </div>
              </div>
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(i);
              }}
              style={{
                padding: "0 12px",
                background: "transparent",
                border: "none",
                color: "var(--text-dim)",
                cursor: "pointer",
                transition: "color 0.15s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#f87171")}
              onMouseLeave={(e) =>
                (e.currentTarget.style.color = "var(--text-dim)")
              }
            >
              <X size={13} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ─── EditPanel ───────────────────────────────────────────────────────────────

interface EditPanelProps {
  onReset: () => void;
  climb: NamedHoldset | null;
  gradeOptions: string[];
  onUpdateClimb: (u: Partial<NamedHoldset>) => void;
  displaySettings: DisplaySettings;
}

function EditPanel({
  onReset,
  climb,
  gradeOptions,
  onUpdateClimb,
  displaySettings,
}: EditPanelProps) {
  const holdset = climb?.holdset ?? null;
  const holdCounts = useMemo(() => {
    if (!holdset) return { hand: 0, foot: 0, start: 0, finish: 0 };
    return {
      hand: holdset.hand.length,
      foot: holdset.foot.length,
      start: holdset.start.length,
      finish: holdset.finish.length,
    };
  }, [holdset]);

  const panelInput: React.CSSProperties = {
    width: "100%",
    background: "var(--bg)",
    color: "var(--text-primary)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "6px 10px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.7rem",
    outline: "none",
    textAlign: "center" as const,
  };
  const actionBtn: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    width: "100%",
    padding: "9px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.6rem",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    background: "transparent",
    border: "1px solid var(--border)",
    color: "var(--text-muted)",
    cursor: "pointer",
    borderRadius: "var(--radius)",
    transition: "all 0.15s",
  };

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: "var(--surface)",
        borderLeft: "1px solid var(--border)",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div
            style={{ width: "2px", height: "14px", background: "var(--cyan)" }}
          />
          <span
            className="bz-oswald"
            style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}
          >
            Edit Climb
          </span>
        </div>
      </div>

      {holdset && (
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "16px",
            display: "flex",
            flexDirection: "column",
            gap: "14px",
          }}
        >
          {/* Name / grade card */}
          <div
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "14px 12px",
              textAlign: "center",
            }}
          >
            <div
              style={{ display: "flex", flexDirection: "column", gap: "8px" }}
            >
              <input
                type="text"
                value={climb?.name || ""}
                placeholder="Climb Name"
                onChange={(e) => onUpdateClimb({ name: e.target.value })}
                style={panelInput}
              />
              <select
                value={climb?.grade || ""}
                onChange={(e) => onUpdateClimb({ grade: e.target.value })}
                style={panelInput}
              >
                {gradeOptions.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Category breakdown */}
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {(["start", "hand", "foot", "finish"] as HoldCategory[]).map(
              (cat) => (
                <div
                  key={cat}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "8px 10px",
                    background: "var(--bg)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                  }}
                >
                  <div
                    style={{
                      width: "8px",
                      height: "8px",
                      borderRadius: "50%",
                      background: displaySettings.categoryColors[cat],
                      flexShrink: 0,
                    }}
                  />
                  <span
                    className="bz-mono"
                    style={{
                      fontSize: "0.65rem",
                      color: "var(--text-muted)",
                      flex: 1,
                    }}
                  >
                    {CATEGORY_LABELS[cat]}
                  </span>
                  <span
                    className="bz-mono"
                    style={{
                      fontSize: "0.65rem",
                      color: "var(--text-primary)",
                    }}
                  >
                    {holdCounts[cat]}
                  </span>
                </div>
              ),
            )}
          </div>

          <>
            <div
              style={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "10px",
              }}
            >
              <p
                className="bz-mono"
                style={{
                  fontSize: "0.6rem",
                  color: "var(--text-primary)",
                  lineHeight: 1.7,
                }}
              >
                Click holds on the layout to cycle through roles. Edit name and
                grade above.
              </p>
            </div>
            <button onClick={onReset} style={actionBtn}>
              <RotateCcw size={10} /> Reset to Generated
            </button>
          </>
        </div>
      )}

      {/* Legend footer */}
      <div
        style={{
          padding: "10px 16px",
          borderTop: "1px solid var(--border)",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px",
        }}
      >
        {(["start", "finish", "hand", "foot"] as HoldCategory[]).map((cat) => (
          <div
            key={cat}
            style={{ display: "flex", alignItems: "center", gap: "6px" }}
          >
            <div
              style={{
                width: "7px",
                height: "7px",
                borderRadius: "50%",
                background: displaySettings.categoryColors[cat],
              }}
            />
            <span
              className="bz-mono"
              style={{ fontSize: "0.55rem", color: "var(--text-muted)" }}
            >
              {CATEGORY_LABELS[cat]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

function SetPage() {
  const navigate = useNavigate();
  const { layoutId: layoutIdParam } = Route.useParams();
  const { climb: climbParam } = Route.useSearch();
  const { layout, loading, waking, error } = useLayout(layoutIdParam);
  if (waking) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          background: "#000000",
        }}
      >
        <WakingScreen />
      </div>
    );
  }
  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          background: "#000000",
          color: "#ffffff",
          fontFamily: "'Space Mono', monospace",
        }}
      >
        <Loader2 size={24} style={{ animation: "spin 1s linear infinite" }} />
      </div>
    );
  }
  if (error || !layout) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          gap: "16px",
          background: "var(--bg)",
          color: "var(--text-muted)",
          fontFamily: "'Space Mono', monospace",
        }}
      >
        <p>{error ?? "layout not found"}</p>
        <button
          onClick={() => navigate({ to: "/" })}
          style={{
            padding: "8px 16px",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--text-primary)",
            cursor: "pointer",
            borderRadius: "var(--radius)",
            fontFamily: "'Space Mono', monospace",
          }}
        >
          Back to Home
        </button>
      </div>
    );
  } else {
    return (
      <MainSetPage
        layout={layout}
        climbParam={climbParam}
        navigate={navigate}
      />
    );
  }
}

// ─── MainSetPage ─────────────────────────────────────────────────────────────

interface MainSetPageProps {
  layout: LayoutDetail;
  climbParam: string | undefined;
  navigate: UseNavigateResult<string>;
}

function MainSetPage({ layout, climbParam, navigate }: MainSetPageProps) {
  const layoutId = layout.metadata.id;
  const layoutDimensions = {
    width: layout.metadata.dimensions[0],
    height: layout.metadata.dimensions[1],
  };

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [gradingScale, setGradingScale] = useState<GradeScale>("v_grade");
  const [gradeOptions, setGradeOptions] = useState(VGRADE_OPTIONS);
  const [numClimbs, setNumClimbs] = useState<number | null>(3);
  const [grade, setGrade] = useState<string>("V4");
  const [angle, setAngle] = useState<number | null>(null);
  const [generateSettings, setGenerateSettings] = useState<GenerateSettings>(
    DEFAULT_GENERATE_SETTINGS,
  );
  const [showModelSettings, setShowModelSettings] = useState(false);
  const [generatedClimbs, setGeneratedClimbs] = useState<NamedHoldset[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>(
    DEFAULT_DISPLAY_SETTINGS,
  );
  const [showDisplaySettings, setShowDisplaySettings] = useState(false);
  const originalHoldsetsRef = useRef<Holdset[]>([]);
  const [isExporting, setIsExporting] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);
  const hasNativeShare = typeof navigator !== "undefined" && !!navigator.share;
  const [mobilePanel, setMobilePanel] = useState<"none" | "left" | "right">(
    "none",
  );
  const closeMobilePanel = useCallback(() => setMobilePanel("none"), []);
  const selectedClimb =
    selectedIndex !== null ? generatedClimbs[selectedIndex] : null;
  const selectedHoldset = selectedClimb?.holdset ?? null;

  // Size state
  const [activeSize, setActiveSize] = useState<SizeMetadata | null>(() => {
    const sizes = layout.metadata.sizes;
    return sizes.find((s) => s.name === "default") ?? sizes[0] ?? null;
  });
  const [showSizeDropdown, setShowSizeDropdown] = useState(false);

  const handleSelectSize = useCallback((size: SizeMetadata) => {
    setActiveSize(size);
    setShowSizeDropdown(false);
    setGeneratedClimbs([]);
    originalHoldsetsRef.current = [];
    setSelectedIndex(null);
  }, []);

  // Auth state
  const { isSignedIn, user } = useUser();
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Reset save success when switching climbs
  useEffect(() => {
    setSaveSuccess(false);
  }, [selectedIndex]);

  // Decode shared climb from URL
  useEffect(() => {
    if (!climbParam) return;
    const decoded = decodeClimbFromParam(climbParam);
    if (!decoded) return;
    setGeneratedClimbs([decoded]);
    originalHoldsetsRef.current = [
      {
        start: [...decoded.holdset.start],
        finish: [...decoded.holdset.finish],
        hand: [...decoded.holdset.hand],
        foot: [...decoded.holdset.foot],
      },
    ];
    setSelectedIndex(0);
  }, []);

  const handleImageLoad = useCallback(
    (d: { width: number; height: number }) => setImageDimensions(d),
    [],
  );

  const handleGenerate = useCallback(async () => {
    try {
      setIsGenerating(true);
      setError(null);
      const generate_grade = grade ?? gradeOptions[0];
      const request: GenerateRequest = {
        num_climbs: numClimbs ?? 3,
        grade: generate_grade,
        grade_scale: gradingScale,
        angle: angle ?? null,
      };
      const response = await generateClimbs(
        layoutId,
        request,
        generateSettings,
      );
      const named: NamedHoldset[] = response.climbs.map((holdset) => ({
        name: generateClimbName(),
        grade: generate_grade,
        angle: request.angle?.toString() ?? "40",
        holdset,
      }));
      setGeneratedClimbs((prev) => [...named, ...prev]);
      originalHoldsetsRef.current = [
        ...response.climbs.map((h) => ({
          start: [...h.start],
          finish: [...h.finish],
          hand: [...h.hand],
          foot: [...h.foot],
        })),
        ...originalHoldsetsRef.current,
      ];
      if (response.climbs.length > 0) setSelectedIndex(0);
      navigate({
        to: "/$layoutId/set",
        params: { layoutId: layoutId },
        search: {},
        replace: true,
      });
      setIsGenerating(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
      setIsGenerating(false);
    }
  }, [
    layoutId,
    numClimbs,
    grade,
    gradingScale,
    gradeOptions,
    generateSettings,
    angle,
    navigate,
  ]);

  const handleGradingScaleChange = useCallback((scale: GradeScale) => {
    if (scale === "v_grade") {
      setGradingScale("v_grade");
      setGradeOptions(VGRADE_OPTIONS);
      setGrade("V0");
    } else {
      setGradingScale("font");
      setGradeOptions(FONT_OPTIONS);
      setGrade("4a");
    }
  }, []);

  const handleUpdateClimb = useCallback(
    (updates: Partial<NamedHoldset>) => {
      setGeneratedClimbs((prev) => {
        if (selectedIndex === null) return prev;
        return prev.map((climb, i) =>
          i === selectedIndex ? { ...climb, ...updates } : climb,
        );
      });
    },
    [selectedIndex],
  );

  const handleResetHoldset = useCallback(() => {
    if (selectedIndex === null) return;
    const original = originalHoldsetsRef.current[selectedIndex];
    if (!original) return;
    setGeneratedClimbs((prev) =>
      prev.map((entry, i) =>
        i === selectedIndex
          ? {
              ...entry,
              holdset: {
                start: [...original.start],
                finish: [...original.finish],
                hand: [...original.hand],
                foot: [...original.foot],
              },
            }
          : entry,
      ),
    );
  }, [selectedIndex]);

  const handleDeleteClimb = useCallback((index: number) => {
    setGeneratedClimbs((prev) => prev.filter((_, i) => i !== index));
    originalHoldsetsRef.current = originalHoldsetsRef.current.filter(
      (_, i) => i !== index,
    );
    setSelectedIndex((current) => {
      if (current === null) return null;
      if (current >= index && current > 0) return current - 1;
      return current;
    });
  }, []);

  const handleClearClimbs = useCallback(() => {
    setGeneratedClimbs([]);
    originalHoldsetsRef.current = [];
    setSelectedIndex(null);
  }, []);

  const handleCopyLink = useCallback(() => {
    if (!selectedClimb) return;
    navigator.clipboard
      .writeText(buildShareUrl(layoutId, selectedClimb))
      .then(() => {
        setLinkCopied(true);
        setTimeout(() => setLinkCopied(false), 2000);
      });
  }, [layoutId, selectedClimb]);

  const handleExportImage = useCallback(async () => {
    if (!selectedClimb) return;
    setIsExporting(true);
    try {
      const blob = await renderExportImage(
        layoutId,
        layout.metadata.name,
        layout.holds ?? [],
        layoutDimensions,
        selectedClimb,
        user?.fullName ?? null,
        displaySettings,
        layout.metadata.image_edges as [number, number, number, number] | null,
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${selectedClimb.name.replace(/\s+/g, "_")}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setIsExporting(false);
    }
  }, [layoutId, layout, layoutDimensions, selectedClimb, displaySettings]);

  const handleNativeShare = useCallback(async () => {
    if (!selectedClimb) return;
    try {
      const url = buildShareUrl(layoutId, selectedClimb);
      let file: File | undefined;
      try {
        const blob = await renderExportImage(
          layoutId,
          layout.metadata.name,
          layout.holds ?? [],
          layoutDimensions,
          selectedClimb,
          user?.fullName ?? null,
          displaySettings,
          layout.metadata.image_edges as
            | [number, number, number, number]
            | null,
        );
        file = new File(
          [blob],
          `${selectedClimb.name.replace(/\s+/g, "_")}.png`,
          { type: "image/png" },
        );
      } catch {}
      const shareData: ShareData = {
        title: selectedClimb.name,
        text: `Check out this climb: ${selectedClimb.name}`,
        url,
      };
      if (file && navigator.canShare?.({ files: [file] }))
        shareData.files = [file];
      await navigator.share(shareData);
    } catch (err) {
      if ((err as Error).name !== "AbortError") handleCopyLink();
    }
  }, [
    layoutId,
    layout,
    layoutDimensions,
    selectedClimb,
    displaySettings,
    handleCopyLink,
  ]);

  const handleCreateBlank = useCallback(() => {
    const blankHoldset: Holdset = { start: [], finish: [], hand: [], foot: [] };
    const blankClimb: NamedHoldset = {
      name: generateClimbName(),
      grade: grade ?? "V?",
      angle: (angle ?? 40).toString(),
      holdset: blankHoldset,
    };
    setGeneratedClimbs((prev) => [blankClimb, ...prev]);
    originalHoldsetsRef.current = [
      { start: [], finish: [], hand: [], foot: [] },
      ...originalHoldsetsRef.current,
    ];
    setSelectedIndex(0);
  }, [grade, angle]);

  const handleSaveToDatabase = useCallback(async () => {
    if (!selectedClimb || !isSignedIn) return;
    setIsSaving(true);
    setSaveSuccess(false);
    setSaveError(null);
    try {
      await createClimb(layoutId, {
        name: selectedClimb.name,
        holdset: selectedClimb.holdset,
        scale: gradingScale,
        grade: selectedClimb.grade,
        angle: parseInt(selectedClimb.angle),
        setter_name: user.fullName!,
        setter_id: user.id,
        tags: [],
      });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      const status = (err as { response?: { status?: number } }).response
        ?.status;
      if (status === 409) {
        setSaveError("Already saved");
        setTimeout(() => setSaveError(null), 3000);
      } else {
        console.error("Failed to save climb:", err);
        setError(
          err instanceof Error
            ? err.message
            : "Failed to save climb to database",
        );
      }
    } finally {
      setIsSaving(false);
    }
  }, [layoutId, selectedClimb, isSignedIn]);

  const handleHoldClick = useCallback(
    (holdIndex: number) => {
      if (selectedIndex === null) return;
      // Bounds check: ignore clicks on holds outside the active size window
      if (activeSize && activeSize.name !== "default") {
        const hold = layout.holds.find((h) => h.hold_index === holdIndex);
        if (hold) {
          const [leftFt, rightFt, bottomFt, topFt] = activeSize.edges;
          if (
            hold.x < leftFt ||
            hold.x > rightFt ||
            hold.y < bottomFt ||
            hold.y > topFt
          )
            return;
        }
      }
      setGeneratedClimbs((prev) => {
        const entry = prev[selectedIndex];
        if (!entry) return prev;
        const holdset = entry.holdset;
        let currentCat: HoldCategory | null = null;
        if (holdset.start.includes(holdIndex)) currentCat = "start";
        else if (holdset.finish.includes(holdIndex)) currentCat = "finish";
        else if (holdset.hand.includes(holdIndex)) currentCat = "hand";
        else if (holdset.foot.includes(holdIndex)) currentCat = "foot";
        const removeFromAll = (hs: Holdset, idx: number): Holdset => ({
          start: hs.start.filter((h) => h !== idx),
          finish: hs.finish.filter((h) => h !== idx),
          hand: hs.hand.filter((h) => h !== idx),
          foot: hs.foot.filter((h) => h !== idx),
        });
        let newHoldset: Holdset;
        if (currentCat === null) {
          newHoldset = { ...holdset, hand: [...holdset.hand, holdIndex] };
        } else {
          const currentIndex = CATEGORY_ORDER.indexOf(currentCat);
          let nextIndex = currentIndex + 1;
          const cleaned = removeFromAll(holdset, holdIndex);
          while (nextIndex < CATEGORY_ORDER.length) {
            const nextCat = CATEGORY_ORDER[nextIndex];
            if (nextCat === "start" && cleaned.start.length >= 2) {
              nextIndex++;
              continue;
            }
            if (nextCat === "finish" && cleaned.finish.length >= 2) {
              nextIndex++;
              continue;
            }
            break;
          }
          if (nextIndex >= CATEGORY_ORDER.length) {
            newHoldset = cleaned;
          } else {
            const nextCat = CATEGORY_ORDER[nextIndex];
            newHoldset = {
              ...cleaned,
              [nextCat]: [...cleaned[nextCat], holdIndex],
            };
          }
        }
        return prev.map((e, i) =>
          i === selectedIndex ? { ...e, holdset: newHoldset } : e,
        );
      });
    },
    [selectedIndex, activeSize, layout.holds],
  );

  const handleSelectClimb = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const handleSwipeNext = useCallback(() => {
    if (generatedClimbs.length === 0) return;
    setSelectedIndex((prev) =>
      prev === null ? 0 : (prev + 1) % generatedClimbs.length,
    );
  }, [generatedClimbs.length]);

  const handleSwipePrev = useCallback(() => {
    if (generatedClimbs.length === 0) return;
    setSelectedIndex((prev) =>
      prev === null
        ? generatedClimbs.length - 1
        : (prev - 1 + generatedClimbs.length) % generatedClimbs.length,
    );
  }, [generatedClimbs.length]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      )
        return;
      if (e.key === "ArrowLeft") {
        if (generatedClimbs.length > 1) handleSwipePrev();
      } else if (e.key === "ArrowRight") {
        if (generatedClimbs.length > 1) handleSwipeNext();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [generatedClimbs.length, handleSwipeNext, handleSwipePrev]);

  // Shared panel props
  const generationPanelProps = {
    displaySettings,
    gradingScale,
    gradeOptions,
    grade,
    onGradingScaleChange: handleGradingScaleChange,
    onGradeChange: setGrade,
    numClimbs,
    onNumClimbsChange: setNumClimbs,
    angle,
    angleFixed: false,
    onAngleChange: setAngle,
    generateSettings,
    onGenerateSettingsChange: setGenerateSettings,
    showModelSettings,
    onToggleModelSettings: () => setShowModelSettings((v) => !v),
    isGenerating,
    error,
    onGenerate: handleGenerate,
    onCreateBlank: handleCreateBlank,
    holdsets: generatedClimbs,
    selectedIndex,
    onSelectHoldset: handleSelectClimb,
    onDeleteHoldset: handleDeleteClimb,
    onClearHoldsets: handleClearClimbs,
  };

  const editPanelProps = {
    onReset: handleResetHoldset,
    climb: selectedClimb,
    gradeOptions,
    onUpdateClimb: handleUpdateClimb,
    displaySettings,
  };

  const saveShareProps: SaveShareMenuProps = {
    onCopyLink: handleCopyLink,
    onExportImage: handleExportImage,
    onNativeShare: handleNativeShare,
    onSaveToDatabase: handleSaveToDatabase,
    isExporting,
    isSaving,
    linkCopied,
    hasNativeShare,
    isSignedIn: !!isSignedIn,
    hasClimb: !!selectedClimb,
    saveSuccess,
    saveError,
  };

  return (
    <>
      <style>{GLOBAL_STYLES}</style>

      <div
        style={{
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          background: "var(--bg)",
          color: "var(--text-primary)",
          position: "relative",
        }}
      >
        {/* Header */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 20px",
            height: "48px",
            flexShrink: 0,
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
            zIndex: 20,
          }}
        >
          {/* Left */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <button
              onClick={() => navigate({ to: "/" })}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "5px",
                background: "transparent",
                border: "none",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.65rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
                cursor: "pointer",
                transition: "color 0.15s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.color = "var(--cyan)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.color = "var(--text-muted)")
              }
            >
              <ArrowLeft size={12} />
              <span className="hidden sm:inline">Back</span>
            </button>
            <div
              style={{
                width: "1px",
                height: "16px",
                background: "var(--border)",
              }}
            />
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div
                style={{
                  width: "2px",
                  height: "14px",
                  background: "var(--cyan)",
                }}
              />
              <span
                className="bz-oswald"
                style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}
              >
                {layout.metadata.name}
              </span>
            </div>
            {isSignedIn && user?.id === layout.metadata.owner_id && (
              <>
                <div
                  style={{
                    width: "1px",
                    height: "16px",
                    background: "var(--border)",
                  }}
                />
                <button
                  onClick={() =>
                    navigate({
                      to: "/$layoutId/holds",
                      params: { layoutId: layoutId },
                    })
                  }
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "5px",
                    background: "transparent",
                    border: "none",
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.65rem",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: "var(--text-muted)",
                    cursor: "pointer",
                    transition: "color 0.15s",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.color = "var(--cyan)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.color = "var(--text-muted)")
                  }
                >
                  <Pencil size={12} />
                  <span className="hidden sm:inline">Edit Holds</span>
                </button>
              </>
            )}
          </div>

          {/* Center */}
          <button
            className="bz-mono"
            style={{
              fontSize: "0.55rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--text-primary)",
            }}
            onClick={() =>
              navigate({
                to: "/$layoutId/view",
                params: { layoutId: layoutId },
              })
            }
          >
            Climbs
          </button>
          <a
            href="https://docs.google.com/forms/d/e/1FAIpQLSeYDIel5MMjj0X3zlXFe4N4FZdUcXadAL5bR-Wjb4W7SVZ5SQ/viewform?usp=dialog"
            target="_blank"
            rel="noopener noreferrer"
            className="bz-mono"
            style={{
              fontSize: "0.55rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--text-dim)",
            }}
          >
            Give Feedback
          </a>

          {/* Right: sizes link + display settings */}
          <div
            style={{
              position: "relative",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            {/* Size picker dropdown */}
            <div style={{ position: "relative" }}>
              <button
                onClick={() => setShowSizeDropdown((v) => !v)}
                title="Select Size"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                  padding: "0 8px",
                  height: "28px",
                  border: `1px solid ${showSizeDropdown || activeSize?.name !== "default" ? "var(--border-active)" : "var(--border)"}`,
                  borderRadius: "var(--radius)",
                  background:
                    showSizeDropdown || activeSize?.name !== "default"
                      ? "var(--cyan-dim)"
                      : "transparent",
                  color:
                    showSizeDropdown || activeSize?.name !== "default"
                      ? "var(--cyan)"
                      : "var(--text-muted)",
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                <Layers size={13} />
                <span
                  className="bz-mono"
                  style={{ fontSize: "0.55rem", letterSpacing: "0.08em" }}
                >
                  {activeSize?.name ?? "—"}
                </span>
                <ChevronDown
                  size={10}
                  style={{
                    transform: showSizeDropdown ? "rotate(180deg)" : "none",
                    transition: "transform 0.15s",
                  }}
                />
              </button>

              {showSizeDropdown && (
                <>
                  <div
                    style={{ position: "fixed", inset: 0, zIndex: 40 }}
                    onClick={() => setShowSizeDropdown(false)}
                  />
                  <div
                    style={{
                      position: "absolute",
                      right: 0,
                      top: "calc(100% + 8px)",
                      background: "var(--surface)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius)",
                      boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
                      zIndex: 50,
                      minWidth: "180px",
                      animation: "bzFadeUp 0.15s ease-out",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        padding: "7px 12px",
                        borderBottom: "1px solid var(--border)",
                      }}
                    >
                      <span
                        className="bz-mono"
                        style={{
                          fontSize: "0.55rem",
                          letterSpacing: "0.12em",
                          textTransform: "uppercase",
                          color: "var(--text-dim)",
                        }}
                      >
                        Active Size
                      </span>
                    </div>
                    {layout.metadata.sizes.map((size) => {
                      const isActive = activeSize?.id === size.id;
                      return (
                        <button
                          key={size.id}
                          onClick={() => handleSelectSize(size)}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            width: "100%",
                            padding: "9px 12px",
                            background: isActive
                              ? "var(--cyan-dim)"
                              : "transparent",
                            border: "none",
                            borderBottom: "1px solid var(--border)",
                            color: isActive
                              ? "var(--cyan)"
                              : "var(--text-primary)",
                            cursor: "pointer",
                            textAlign: "left",
                            transition: "background 0.1s",
                          }}
                        >
                          <div
                            style={{
                              width: "6px",
                              height: "6px",
                              borderRadius: "50%",
                              background: isActive
                                ? "var(--cyan)"
                                : "var(--border)",
                              flexShrink: 0,
                            }}
                          />
                          <span
                            className="bz-mono"
                            style={{ fontSize: "0.65rem", flex: 1 }}
                          >
                            {size.name}
                          </span>
                          {size.kickboard && (
                            <span
                              className="bz-mono"
                              style={{
                                fontSize: "0.55rem",
                                color: "var(--text-dim)",
                              }}
                            >
                              KB
                            </span>
                          )}
                        </button>
                      );
                    })}
                    <button
                      onClick={() => {
                        setShowSizeDropdown(false);
                        navigate({
                          to: "/$layoutId/sizes",
                          params: { layoutId },
                        });
                      }}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        width: "100%",
                        padding: "8px 12px",
                        background: "transparent",
                        border: "none",
                        color: "var(--text-dim)",
                        cursor: "pointer",
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "0.55rem",
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        transition: "color 0.1s",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.color = "var(--text-muted)")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.color = "var(--text-dim)")
                      }
                    >
                      Manage Sizes →
                    </button>
                  </div>
                </>
              )}
            </div>
            <span
              className="hidden lg:flex bz-mono"
              style={{
                fontSize: "0.55rem",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--text-dim)",
              }}
            >
              Hold Display
            </span>
            <button
              onClick={() => setShowDisplaySettings(!showDisplaySettings)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: "28px",
                height: "28px",
                border: `1px solid ${showDisplaySettings ? "var(--cyan)" : "var(--border)"}`,
                background: showDisplaySettings
                  ? "var(--cyan-dim)"
                  : "transparent",
                color: showDisplaySettings
                  ? "var(--cyan)"
                  : "var(--text-muted)",
                cursor: "pointer",
                borderRadius: "var(--radius)",
                transition: "all 0.15s",
              }}
            >
              <SunMedium size={13} />
            </button>

            {showDisplaySettings && (
              <>
                <div
                  style={{ position: "fixed", inset: 0, zIndex: 40 }}
                  onClick={() => setShowDisplaySettings(false)}
                />
                <div
                  style={{
                    position: "absolute",
                    right: 0,
                    top: "calc(100% + 8px)",
                    padding: "16px",
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
                    zIndex: 50,
                    width: "280px",
                    animation: "bzFadeUp 0.15s ease-out",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginBottom: "16px",
                    }}
                  >
                    <div
                      style={{
                        width: "2px",
                        height: "10px",
                        background: "var(--cyan)",
                      }}
                    />
                    <SectionLabel>Hold Display Settings</SectionLabel>
                  </div>
                  <DisplaySettingsPanel
                    settings={displaySettings}
                    onChange={setDisplaySettings}
                  />
                </div>
              </>
            )}
          </div>
        </header>

        {/* Mobile climb chip */}
        {selectedClimb && (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              position: "absolute",
              top: "56px",
              left: 0,
              right: 0,
              zIndex: 10,
              pointerEvents: "none",
            }}
            className="lg:hidden"
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                pointerEvents: "auto",
                animation: "bzFadeUp 0.2s ease-out",
              }}
            >
              {generatedClimbs.length > 1 && (
                <button
                  onClick={handleSwipePrev}
                  style={{
                    background: "rgba(17,17,19,0.92)",
                    backdropFilter: "blur(8px)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    width: "36px",
                    height: "36px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--text-primary)",
                    cursor: "pointer",
                  }}
                >
                  <ChevronLeft size={18} />
                </button>
              )}
              <div
                style={{
                  background: "rgba(17,17,19,0.92)",
                  backdropFilter: "blur(8px)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  padding: "8px 18px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                }}
              >
                <span
                  className="bz-oswald text-center text-[1.6rem] lg:text-[3rem]"
                  style={{ color: "var(--text-primary)" }}
                >
                  {selectedClimb.name}
                </span>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginTop: "2px",
                  }}
                >
                  <span
                    className="bz-mono text-[0.6rem] lg:text-[0.8rem]"
                    style={{ fontSize: "0.6rem", color: "var(--cyan)" }}
                  >
                    {selectedClimb.grade} | {selectedClimb.angle}°
                  </span>
                  <span
                    className="bz-mono"
                    style={{ fontSize: "0.6rem", color: "var(--ruby)" }}
                  ></span>
                  {generatedClimbs.length > 1 && selectedIndex !== null && (
                    <span
                      className="bz-mono"
                      style={{ fontSize: "0.6rem", color: "var(--text-dim)" }}
                    >
                      {selectedIndex + 1}/{generatedClimbs.length}
                    </span>
                  )}
                </div>
              </div>
              {generatedClimbs.length > 1 && (
                <button
                  onClick={handleSwipeNext}
                  style={{
                    background: "rgba(17,17,19,0.92)",
                    backdropFilter: "blur(8px)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    width: "36px",
                    height: "36px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--text-primary)",
                    cursor: "pointer",
                  }}
                >
                  <ChevronRight size={18} />
                </button>
              )}
            </div>
          </div>
        )}

        {/* Body */}
        <div
          style={{
            flex: 1,
            display: "flex",
            minHeight: 0,
            position: "relative",
          }}
        >
          {/* Left panel (desktop) */}
          <div
            style={{
              width: "300px",
              flexShrink: 0,
              flexDirection: "column",
              borderRight: "1px solid var(--border)",
              background: "var(--surface)",
            }}
            className="hidden lg:flex"
          >
            <GenerationPanel {...generationPanelProps} />
          </div>

          {/* Mobile left drawer */}
          {mobilePanel === "left" && (
            <div
              style={{
                position: "fixed",
                inset: 0,
                zIndex: 40,
                display: "flex",
              }}
              className="lg:hidden"
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.7)",
                }}
                onClick={closeMobilePanel}
              />
              <div
                style={{
                  position: "relative",
                  width: "300px",
                  maxWidth: "85vw",
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  background: "var(--surface)",
                  borderRight: "1px solid var(--border)",
                  zIndex: 10,
                  animation: "bzSlideInLeft 0.2s ease-out",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "12px 16px",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                    }}
                  >
                    <div
                      style={{
                        width: "2px",
                        height: "12px",
                        background: "var(--cyan)",
                      }}
                    />
                    <span
                      className="bz-oswald"
                      style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}
                    >
                      Climbs & Generation
                    </span>
                  </div>
                  <button
                    onClick={closeMobilePanel}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--text-muted)",
                      cursor: "pointer",
                    }}
                  >
                    <X size={16} />
                  </button>
                </div>
                <div
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    minHeight: 0,
                    overflow: "hidden",
                  }}
                >
                  <GenerationPanel
                    {...generationPanelProps}
                    onSelectHoldset={(i) => {
                      handleSelectClimb(i);
                      closeMobilePanel();
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Canvas */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <WallCanvas
              layoutId={layoutId}
              holds={layout.holds ?? []}
              wallDimensions={layoutDimensions}
              selectedHoldset={selectedHoldset}
              imageDimensions={imageDimensions}
              onImageLoad={handleImageLoad}
              displaySettings={displaySettings}
              onHoldClick={handleHoldClick}
              activeSize={activeSize}
              imageEdges={
                layout.metadata.image_edges as
                  | [number, number, number, number]
                  | null
              }
              homographySrcCorners={layout.metadata.homography_src_corners}
            />
          </div>

          {/* Right panel (desktop) */}
          <div
            style={{
              width: "260px",
              flexShrink: 0,
              flexDirection: "column",
            }}
            className="hidden lg:flex"
          >
            <div
              style={{
                flex: 1,
                minHeight: 0,
                overflow: "hidden",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <EditPanel {...editPanelProps} />
            </div>
            <DesktopSaveSharePanel {...saveShareProps} />
          </div>

          {/* Mobile right drawer */}
          {mobilePanel === "right" && (
            <div
              style={{
                position: "fixed",
                inset: 0,
                zIndex: 40,
                display: "flex",
                justifyContent: "flex-end",
              }}
              className="lg:hidden"
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.7)",
                }}
                onClick={closeMobilePanel}
              />
              <div
                style={{
                  position: "relative",
                  width: "300px",
                  maxWidth: "85vw",
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  background: "var(--surface)",
                  borderLeft: "1px solid var(--border)",
                  zIndex: 10,
                  animation: "bzSlideInRight 0.2s ease-out",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "12px 16px",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                    }}
                  >
                    <div
                      style={{
                        width: "2px",
                        height: "12px",
                        background: "var(--cyan)",
                      }}
                    />
                    <span
                      className="bz-oswald"
                      style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}
                    >
                      Edit Climb
                    </span>
                  </div>
                  <button
                    onClick={closeMobilePanel}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--text-muted)",
                      cursor: "pointer",
                    }}
                  >
                    <X size={16} />
                  </button>
                </div>
                <div
                  style={{
                    flex: 1,
                    overflowY: "auto",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <EditPanel {...editPanelProps} />
                </div>
              </div>
            </div>
          )}

          {/* Mobile FABs - climb navigation */}
          <MobileSwipeNav
            count={generatedClimbs.length}
            onPrev={handleSwipePrev}
            onNext={handleSwipeNext}
          />

          {/* Mobile FABs - main actions */}
          <div
            style={{
              position: "absolute",
              bottom: "48px",
              left: 0,
              right: 0,
              justifyContent: "center",
              gap: "10px",
              zIndex: 30,
              padding: "0 16px",
            }}
            className="flex lg:hidden"
          >
            <button
              onClick={() =>
                setMobilePanel((p) => (p === "left" ? "none" : "left"))
              }
              style={{
                pointerEvents: "auto",
                display: "flex",
                alignItems: "center",
                gap: "7px",
                padding: "10px 18px",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.65rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                cursor: "pointer",
                borderRadius: "var(--radius)",
                boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
              }}
            >
              <Sparkles size={12} style={{ color: "var(--cyan)" }} />
              {generatedClimbs.length > 0 ? "Climbs" : "Generate"}
            </button>

            <button
              onClick={() =>
                setMobilePanel((p) => (p === "right" ? "none" : "right"))
              }
              style={{
                pointerEvents: "auto",
                display: "flex",
                alignItems: "center",
                gap: "7px",
                padding: "10px 18px",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.65rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                cursor: "pointer",
                borderRadius: "var(--radius)",
                boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
              }}
            >
              <Pencil size={12} /> Edit
            </button>

            <SaveShareMenu {...saveShareProps} />
          </div>
        </div>
      </div>
    </>
  );
}
