import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { generateClimbs } from "@/api/generate";
import { WakingScreen } from "@/components";
import {
  ArrowLeft,
  Sparkles,
  Loader2,
  Pencil,
  RotateCcw,
  Link,
  Check,
  Share2,
  Image,
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
} from "lucide-react";
import type {
  WallDetail,
  HoldDetail,
  Holdset,
  GenerateRequest,
  GenerateSettings,
  GradeScale,
} from "@/types";
import {
  DEFAULT_GENERATE_SETTINGS,
  FAST_GENERATE_SETTINGS,
  SLOW_GENERATE_SETTINGS,
} from "@/types";
import { is502 } from "@/api";

// ─── Design tokens (matches HomePage) ───────────────────────────────────────
const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&family=Space+Mono:wght@400;700&display=swap');

  :root {
    --cyan: #06b6d4;
    --cyan-dim: rgba(6,182,212,0.15);
    --cyan-glow: rgba(6,182,212,0.25);
    --ruby: #5a0e15;
    --bg: #09090b;
    --surface: #111113;
    --surface2: #18181b;
    --border: rgba(255,255,255,0.08);
    --border-active: #06b6d4;
    --text-primary: #f4f4f5;
    --text-muted: #71717a;
    --text-dim: #3f3f46;
    --radius: 4px;
  }

  /* Typography helpers */
  .bz-oswald {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .bz-mono {
    font-family: 'Space Mono', monospace;
  }

  /* Range input reset */
  .bz-range {
    -webkit-appearance: none;
    width: 100%;
    height: 2px;
    background: rgba(255,255,255,0.1);
    border-radius: 0;
    cursor: pointer;
  }
  .bz-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    background: var(--cyan);
    border-radius: 0;
    cursor: pointer;
  }
  .bz-range::-moz-range-thumb {
    width: 12px;
    height: 12px;
    background: var(--cyan);
    border-radius: 0;
    border: none;
    cursor: pointer;
  }

  /* Slide-in animations for mobile drawers */
  @keyframes bzSlideInLeft {
    from { transform: translateX(-100%); }
    to   { transform: translateX(0); }
  }
  @keyframes bzSlideInRight {
    from { transform: translateX(100%); }
    to   { transform: translateX(0); }
  }
  @keyframes bzFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes bzFadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
`;

// ─── Route ───────────────────────────────────────────────────────────────────

interface ClimbSearchParams {
  climb?: string;
}

export const Route = createFileRoute("/$wallId")({
  component: GeneratePage,
  validateSearch: (search: Record<string, unknown>): ClimbSearchParams => ({
    climb: typeof search.climb === "string" ? search.climb : undefined,
  }),
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
  staleTime: 3_600_000,
});

// ─── Hold category constants ──────────────────────────────────────────────────

type HoldCategory = "hand" | "foot" | "start" | "finish";
const CATEGORY_ORDER: HoldCategory[] = ["hand", "foot", "start", "finish"];
const CATEGORY_COLORS: Record<HoldCategory, string> = {
  hand: "#3b82f6",
  foot: "#a855f7",
  start: "#22c55e",
  finish: "#ffea00",
};
const CATEGORY_LABELS: Record<HoldCategory, string> = {
  hand: "Hand",
  foot: "Foot",
  start: "Start",
  finish: "Finish",
};
const HOLD_STROKE_COLOR = "#00b679";

// ─── Display settings ─────────────────────────────────────────────────────────

type ColorMode = "role" | "uniform";
interface DisplaySettings {
  scale: number;
  colorMode: ColorMode;
  uniformColor: string;
  categoryColors: Record<HoldCategory, string>;
  opacity: number;
  filled: boolean;
}
const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  scale: 1.0,
  colorMode: "role",
  uniformColor: HOLD_STROKE_COLOR,
  categoryColors: { ...CATEGORY_COLORS },
  opacity: 0.6,
  filled: true,
};

// ─── Grade options ────────────────────────────────────────────────────────────

const VGRADE_OPTIONS = [
  "V0-",
  "V0",
  "V0+",
  "V1",
  "V1+",
  "V2",
  "V2+",
  "V3",
  "V3+",
  "V4",
  "V4+",
  "V5",
  "V5+",
  "V6",
  "V6+",
  "V7",
  "V7+",
  "V8",
  "V8+",
  "V9",
  "V9+",
  "V10",
  "V10+",
  "V11",
  "V11+",
  "V12",
  "V12+",
  "V13",
  "V13+",
  "V14",
  "V14+",
  "V15",
  "V15+",
  "V16",
];
const FONT_OPTIONS = [
  "4a",
  "4b",
  "4c",
  "5a",
  "5b",
  "5c",
  "6a",
  "6a+",
  "6b",
  "6b+",
  "6c",
  "6c+",
  "7a",
  "7a+",
  "7b",
  "7b+",
  "7c",
  "7c+",
  "8a",
  "8a+",
  "8b",
  "8b+",
  "8c",
  "8c+",
];

// ─── Name generator ───────────────────────────────────────────────────────────

const ADJECTIVES = [
  "Angry",
  "Bold",
  "Cosmic",
  "Dancing",
  "Electric",
  "Fierce",
  "Gnarly",
  "Humble",
  "Icy",
  "Jazzy",
  "Killer",
  "Lunar",
  "Mighty",
  "Noble",
  "Obscure",
  "Primal",
  "Quantum",
  "Radical",
  "Savage",
  "Twisted",
  "Ultra",
  "Voluptuous",
  "Wicked",
  "Xtreme",
  "Yonder",
  "Zesty",
];
const ANIMALS = [
  "Aardvark",
  "Badger",
  "Cobra",
  "Dolphin",
  "Eagle",
  "Falcon",
  "Gorilla",
  "Hawk",
  "Ibex",
  "Jaguar",
  "Koala",
  "Lemur",
  "Mantis",
  "Narwhal",
  "Osprey",
  "Panther",
  "Quokka",
  "Raven",
  "Scorpion",
  "Tiger",
  "Urchin",
  "Viper",
  "Wolf",
  "Xingu River Ray",
  "Yak",
  "Zebra",
];
function generateClimbName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const animal = ANIMALS[Math.floor(Math.random() * ANIMALS.length)];
  return `${adj} ${animal}`;
}

// ─── NamedHoldset ─────────────────────────────────────────────────────────────

interface NamedHoldset {
  name: string;
  holdset: Holdset;
  grade: string;
  angle: string;
}

// ─── Sharing utilities ────────────────────────────────────────────────────────

function encodeClimbToParam(entry: NamedHoldset): string {
  const compact = {
    n: entry.name,
    g: entry.grade,
    a: entry.angle,
    s: entry.holdset.start,
    f: entry.holdset.finish,
    h: entry.holdset.hand,
    t: entry.holdset.foot,
  };
  return btoa(JSON.stringify(compact))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}
function decodeClimbFromParam(param: string): NamedHoldset | null {
  try {
    let b64 = param.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4 !== 0) b64 += "=";
    const compact = JSON.parse(atob(b64));
    if (!compact) return null;
    return {
      name: typeof compact.n === "string" ? compact.n : "Unnamed",
      grade: typeof compact.g === "string" ? compact.g : "V?",
      angle: typeof compact.a === "string" ? compact.a : "?",
      holdset: {
        start: Array.isArray(compact.s) ? compact.s : [],
        finish: Array.isArray(compact.f) ? compact.f : [],
        hand: Array.isArray(compact.h) ? compact.h : [],
        foot: Array.isArray(compact.t) ? compact.t : [],
      },
    };
  } catch {
    return null;
  }
}
function buildShareUrl(wallId: string, entry: NamedHoldset): string {
  return `${window.location.origin}/${wallId}?climb=${encodeClimbToParam(entry)}`;
}

// ─── Export canvas renderer ───────────────────────────────────────────────────

async function renderExportImage(
  wallId: string,
  wallName: string,
  holds: HoldDetail[],
  wallDimensions: { width: number; height: number },
  holdset: Holdset,
  climbName: string,
  displaySettings: DisplaySettings,
): Promise<Blob> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new window.Image();
    el.crossOrigin = "anonymous";
    el.onload = () => resolve(el);
    el.onerror = reject;
    el.src = getWallPhotoUrl(wallId);
  });
  const imgW = img.width,
    imgH = img.height;
  const topBannerH = Math.round(imgH * 0.06);
  const bottomBannerH = Math.round(imgH * 0.045);
  const canvas = document.createElement("canvas");
  canvas.width = imgW;
  canvas.height = imgH + topBannerH + bottomBannerH;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111113";
  ctx.fillRect(0, 0, imgW, topBannerH);
  ctx.fillStyle = "#f4f4f5";
  ctx.font = `bold ${Math.round(topBannerH * 0.55)}px sans-serif`;
  ctx.textBaseline = "middle";
  ctx.fillText(climbName, Math.round(imgW * 0.02), topBannerH / 2);
  ctx.fillStyle = "#71717a";
  ctx.font = `${Math.round(topBannerH * 0.4)}px sans-serif`;
  ctx.textAlign = "right";
  ctx.fillText(wallName, imgW - Math.round(imgW * 0.02), topBannerH / 2);
  ctx.textAlign = "left";
  ctx.drawImage(img, 0, topBannerH);
  const startSet = new Set(holdset.start),
    finishSet = new Set(holdset.finish);
  const handSet = new Set(holdset.hand),
    footSet = new Set(holdset.foot);
  const usedSet = new Set([...startSet, ...finishSet, ...handSet, ...footSet]);
  const {
    scale: userScale,
    colorMode,
    uniformColor,
    opacity: userOpacity,
    filled,
  } = displaySettings;
  holds.forEach((hold) => {
    const px = (hold.x / wallDimensions.width) * imgW;
    const py = (1 - hold.y / wallDimensions.height) * imgH + topBannerH;
    const baseScale = imgH / 1000;
    const radius = 10 * baseScale * userScale;
    const isUsed = usedSet.has(hold.hold_index);
    const isStart = startSet.has(hold.hold_index),
      isFinish = finishSet.has(hold.hold_index);
    const isHand = handSet.has(hold.hold_index),
      isFoot = footSet.has(hold.hold_index);
    const baseAlpha = isUsed ? 1 : 0.15;
    const alpha = isUsed ? baseAlpha * userOpacity : baseAlpha;
    let color = HOLD_STROKE_COLOR;
    if (isUsed) {
      if (colorMode === "uniform") color = uniformColor;
      else if (isStart) color = displaySettings.categoryColors.start;
      else if (isFinish) color = displaySettings.categoryColors.finish;
      else if (isHand) color = displaySettings.categoryColors.hand;
      else if (isFoot) color = displaySettings.categoryColors.foot;
    }
    const footScale = isFoot ? 0.5 : 1;
    ctx.beginPath();
    ctx.arc(px, py, radius * footScale, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = isUsed ? baseScale * 2 : 2;
    if (isUsed && filled) {
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
  const legendY = topBannerH + imgH;
  ctx.fillStyle = "#111113";
  ctx.fillRect(0, legendY, imgW, bottomBannerH);
  const legendFont = Math.round(bottomBannerH * 0.45);
  ctx.font = `${legendFont}px sans-serif`;
  ctx.textBaseline = "middle";
  const legendMidY = legendY + bottomBannerH / 2;
  const dotR = Math.round(bottomBannerH * 0.15);
  const pad = Math.round(imgW * 0.02);
  const legendItems = [
    { label: "Start", color: displaySettings.categoryColors.start },
    { label: "Finish", color: displaySettings.categoryColors.finish },
    { label: "Hand", color: displaySettings.categoryColors.hand },
    { label: "Foot", color: displaySettings.categoryColors.foot },
  ];
  let cursorX = pad;
  for (const item of legendItems) {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(cursorX + dotR, legendMidY, dotR, 0, 2 * Math.PI);
    ctx.fill();
    cursorX += dotR * 2 + 6;
    ctx.fillStyle = "#a1a1aa";
    ctx.fillText(item.label, cursorX, legendMidY);
    cursorX += ctx.measureText(item.label).width + pad;
  }
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("toBlob failed"))),
      "image/png",
    );
  });
}

// ─── Shared UI primitives ─────────────────────────────────────────────────────

/** A section label consistent with homepage eyebrow style */
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="bz-mono"
      style={{
        fontSize: "0.6rem",
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: "var(--text-muted)",
      }}
    >
      {children}
    </span>
  );
}

/** Toggle button pair (e.g. By Role / Uniform) */
function TogglePair({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "flex", gap: "2px", width: "100%" }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            flex: value == opt.value ? "1 0 auto" : "1 1 30%",
            padding: "6px 8px",
            minWidth: 0,
            fontSize: "clamp(0.35rem, 2.5vw, 0.65rem)",
            fontFamily: "'Space Mono', monospace",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            overflow: "hidden",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            border: `1px solid ${value === opt.value ? "var(--cyan)" : "var(--border)"}`,
            background: value === opt.value ? "var(--cyan-dim)" : "transparent",
            color: value === opt.value ? "var(--cyan)" : "var(--text-muted)",
            cursor: "pointer",
            transition: "all 0.15s",
            borderRadius: "var(--radius)",
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/** Labeled range slider */
function BzRange({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  leftLabel,
  rightLabel,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  displayValue: string;
  leftLabel?: string;
  rightLabel?: string;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "8px",
        }}
      >
        <SectionLabel>{label}</SectionLabel>
        <span
          className="bz-mono"
          style={{ fontSize: "0.65rem", color: "var(--cyan)" }}
        >
          {displayValue}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="bz-range"
      />
      {(leftLabel || rightLabel) && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "4px",
          }}
        >
          <span
            className="bz-mono"
            style={{ fontSize: "0.55rem", color: "var(--text-dim)" }}
          >
            {leftLabel}
          </span>
          <span
            className="bz-mono"
            style={{ fontSize: "0.55rem", color: "var(--text-dim)" }}
          >
            {rightLabel}
          </span>
        </div>
      )}
    </div>
  );
}

// ─── Display Settings Panel ───────────────────────────────────────────────────

function DisplaySettingsPanel({
  settings,
  onChange,
}: {
  settings: DisplaySettings;
  onChange: (s: DisplaySettings) => void;
}) {
  const update = (patch: Partial<DisplaySettings>) =>
    onChange({ ...settings, ...patch });
  const updateCategoryColor = (cat: HoldCategory, color: string) =>
    update({ categoryColors: { ...settings.categoryColors, [cat]: color } });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        minWidth: "240px",
      }}
    >
      <BzRange
        label="Hold Scale"
        value={settings.scale}
        min={0.3}
        max={3.0}
        step={0.1}
        onChange={(v) => update({ scale: v })}
        displayValue={`${settings.scale.toFixed(1)}×`}
      />

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel>Color Mode</SectionLabel>
        </div>
        <TogglePair
          options={[
            { value: "role", label: "By Role" },
            { value: "uniform", label: "Uniform" },
          ]}
          value={settings.colorMode}
          onChange={(v) => update({ colorMode: v as ColorMode })}
        />
        {settings.colorMode === "uniform" ? (
          <div
            style={{
              marginTop: "10px",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <input
              type="color"
              value={settings.uniformColor}
              onChange={(e) => update({ uniformColor: e.target.value })}
              style={{
                width: "28px",
                height: "28px",
                border: "1px solid var(--border)",
                background: "transparent",
                cursor: "pointer",
                borderRadius: "var(--radius)",
              }}
            />
            <span
              className="bz-mono"
              style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}
            >
              {settings.uniformColor}
            </span>
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "6px",
              marginTop: "10px",
            }}
          >
            {CATEGORY_ORDER.map((cat) => (
              <div
                key={cat}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "6px 8px",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  background: "var(--surface)",
                }}
              >
                <input
                  type="color"
                  value={settings.categoryColors[cat]}
                  onChange={(e) => updateCategoryColor(cat, e.target.value)}
                  style={{
                    width: "18px",
                    height: "18px",
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    padding: 0,
                  }}
                />
                <span
                  className="bz-mono"
                  style={{ fontSize: "0.6rem", color: "var(--text-primary)" }}
                >
                  {CATEGORY_LABELS[cat]}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <BzRange
        label="Opacity"
        value={settings.opacity}
        min={0.1}
        max={1.0}
        step={0.05}
        onChange={(v) => update({ opacity: v })}
        displayValue={`${Math.round(settings.opacity * 100)}%`}
      />

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel>Style</SectionLabel>
        </div>
        <TogglePair
          options={[
            { value: "filled", label: "Filled" },
            { value: "outline", label: "Outline" },
          ]}
          value={settings.filled ? "filled" : "outline"}
          onChange={(v) => update({ filled: v === "filled" })}
        />
      </div>
    </div>
  );
}

// ─── Model Settings Panel ─────────────────────────────────────────────────────

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
    settings.t_start_projection ===
      DEFAULT_GENERATE_SETTINGS.t_start_projection &&
    settings.x_offset === DEFAULT_GENERATE_SETTINGS.x_offset;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <BzRange
        label="Generation Timesteps"
        value={settings.timesteps}
        min={5}
        max={100}
        step={5}
        onChange={(v) => update({ timesteps: v })}
        displayValue={String(settings.timesteps)}
        leftLabel="Faster"
        rightLabel="Higher Quality"
      />
      <BzRange
        label="Projection Start"
        value={settings.t_start_projection}
        min={0.0}
        max={1.0}
        step={0.05}
        onChange={(v) => update({ t_start_projection: v })}
        displayValue={`t=${settings.t_start_projection.toFixed(2)}`}
        leftLabel="Later (Faster)"
        rightLabel="Earlier"
      />

      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: "8px",
          }}
        >
          <SectionLabel>X-Offset</SectionLabel>
          <span
            className="bz-mono"
            style={{ fontSize: "0.65rem", color: "var(--cyan)" }}
          >
            {settings.x_offset != null ? settings.x_offset.toFixed(2) : "Auto"}
          </span>
        </div>
        <input
          type="range"
          min={-1.0}
          max={1.0}
          step={0.05}
          value={settings.x_offset ?? 0}
          onChange={(e) => update({ x_offset: parseFloat(e.target.value) })}
          className="bz-range"
        />
        <button
          onClick={() => update({ x_offset: null })}
          style={{
            marginTop: "8px",
            width: "100%",
            padding: "5px",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            border: `1px solid ${settings.x_offset == null ? "var(--cyan)" : "var(--border)"}`,
            background:
              settings.x_offset == null ? "var(--cyan-dim)" : "transparent",
            color:
              settings.x_offset == null ? "var(--cyan)" : "var(--text-muted)",
            cursor: "pointer",
            borderRadius: "var(--radius)",
            transition: "all 0.15s",
          }}
        >
          {settings.x_offset == null ? "Auto (recommended)" : "Reset to Auto"}
        </button>
      </div>

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel>Generation Style</SectionLabel>
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
          <div>
            <span
              className="bz-mono"
              style={{
                fontSize: "0.6rem",
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
              }}
            >
              generation seed
            </span>
            <input
              type="number"
              style={{
                marginTop: "8px",
                width: "100%",
                padding: "5px",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
                letterSpacing: "0.06em",
                textTransform: "uppercase",
              }}
              value={settings.seed ?? ""}
              onChange={(e) => {
                if (e.target.value != "") {
                  const s = parseInt(e.target.value);
                  if (!isNaN(s)) update({ seed: s });
                }
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

      <p
        className="bz-mono"
        style={{
          fontSize: "0.6rem",
          color: "var(--text-dim)",
          lineHeight: 1.7,
          paddingTop: "10px",
          borderTop: "1px solid var(--border)",
        }}
      >
        Fewer timesteps = faster generation but reduced quality. Projection
        start controls when holds "pull" on the diffusion model. X-Offset shifts
        the climb's center. Auto grid-search is recommended for quality.
      </p>
    </div>
  );
}

// ─── GenerationPanel ──────────────────────────────────────────────────────────

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
  waking: boolean;
  error: string | null;
  onGenerate: () => void;
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
  waking,
  error,
  onGenerate,
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
    generateSettings.t_start_projection === preset.t_start_projection &&
    generateSettings.deterministic === preset.deterministic;
  const isCustom =
    !isPresetActive(FAST_GENERATE_SETTINGS) &&
    !isPresetActive(DEFAULT_GENERATE_SETTINGS) &&
    !isPresetActive(SLOW_GENERATE_SETTINGS);

  // Preset button styles
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

  // Styled select/input
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
      {/* ── Climb Parameters ── */}
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

            {/* Wall angle */}
            <div>
              <div style={{ marginBottom: "6px" }}>
                <SectionLabel>Wall Angle (°)</SectionLabel>
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

      {/* ── Generation Mode ── */}
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
        {waking && <WakingScreen />}
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

      {/* ── Holdset list ── */}
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

// ─── HoldsetList ──────────────────────────────────────────────────────────────

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

// ─── EditPanel ────────────────────────────────────────────────────────────────

interface EditPanelProps {
  editing: boolean;
  onToggleEditing: () => void;
  onReset: () => void;
  onExportImage: () => void;
  onCopyLink: () => void;
  onNativeShare: () => void;
  isExporting: boolean;
  linkCopied: boolean;
  hasNativeShare: boolean;
  climb: NamedHoldset | null;
  gradeOptions: string[];
  onUpdateClimb: (u: Partial<NamedHoldset>) => void;
  displaySettings: DisplaySettings;
}

function EditPanel({
  editing,
  onToggleEditing,
  onReset,
  onExportImage,
  onCopyLink,
  onNativeShare,
  isExporting,
  linkCopied,
  hasNativeShare,
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
        <button
          onClick={onToggleEditing}
          disabled={!holdset}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "5px",
            padding: "5px 10px",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            border: `1px solid ${editing ? "var(--cyan)" : "var(--border)"}`,
            background: editing ? "var(--cyan-dim)" : "transparent",
            color: editing ? "var(--cyan)" : "var(--text-muted)",
            cursor: holdset ? "pointer" : "not-allowed",
            opacity: holdset ? 1 : 0.4,
            borderRadius: "var(--radius)",
            transition: "all 0.15s",
          }}
        >
          <Pencil size={10} /> {editing ? "Done" : "Edit"}
        </button>
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
            {editing ? (
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
            ) : (
              <>
                <div
                  className="bz-oswald"
                  style={{
                    fontSize: "1.1rem",
                    color: "var(--text-primary)",
                    letterSpacing: "0.05em",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {climb?.name}
                </div>
                <div
                  className="bz-mono"
                  style={{
                    fontSize: "0.65rem",
                    color: "var(--cyan)",
                    marginTop: "4px",
                  }}
                >
                  {climb?.grade}
                </div>
              </>
            )}
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

          {editing && (
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
                    color: "var(--text-muted)",
                    lineHeight: 1.7,
                  }}
                >
                  Click holds on the wall to cycle through roles. Edit name and
                  grade above.
                </p>
              </div>
              <button onClick={onReset} style={actionBtn}>
                <RotateCcw size={10} /> Reset to Generated
              </button>
            </>
          )}

          {/* Share section */}
          <div
            style={{
              paddingTop: "10px",
              borderTop: "1px solid var(--border)",
              display: "flex",
              flexDirection: "column",
              gap: "8px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div
                style={{
                  width: "2px",
                  height: "10px",
                  background: "var(--cyan)",
                }}
              />
              <SectionLabel>Share</SectionLabel>
            </div>

            <button onClick={onCopyLink} style={actionBtn}>
              {linkCopied ? (
                <>
                  <Check size={10} style={{ color: "var(--cyan)" }} />
                  <span style={{ color: "var(--cyan)" }}>Link Copied!</span>
                </>
              ) : (
                <>
                  <Link size={10} /> Copy Link
                </>
              )}
            </button>

            <button
              onClick={onExportImage}
              disabled={isExporting}
              style={{
                ...actionBtn,
                opacity: isExporting ? 0.5 : 1,
              }}
            >
              {isExporting ? (
                <>
                  <Loader2
                    size={10}
                    style={{ animation: "spin 1s linear infinite" }}
                  />{" "}
                  Rendering…
                </>
              ) : (
                <>
                  <Image size={10} /> Save Image
                </>
              )}
            </button>

            {hasNativeShare && (
              <button
                onClick={onNativeShare}
                style={{
                  ...actionBtn,
                  background: "var(--cyan)",
                  color: "#09090b",
                  border: "1px solid var(--cyan)",
                  fontWeight: 700,
                }}
              >
                <Share2 size={10} /> Share…
              </button>
            )}
          </div>
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

// ─── WallCanvas ───────────────────────────────────────────────────────────────
// (Unchanged logic; only container background updated to match token)

interface WallCanvasProps {
  wallId: string;
  holds: HoldDetail[];
  wallDimensions: { width: number; height: number };
  selectedHoldset: Holdset | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (d: { width: number; height: number }) => void;
  displaySettings: DisplaySettings;
  editing: boolean;
  onHoldClick: (i: number) => void;
  onSwipeNext?: () => void;
  onSwipePrev?: () => void;
}

function WallCanvas({
  wallId,
  holds,
  wallDimensions,
  selectedHoldset,
  imageDimensions,
  onImageLoad,
  displaySettings,
  editing,
  onHoldClick,
  onSwipeNext,
  onSwipePrev,
}: WallCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [viewTransform, setViewTransform] = useState({ zoom: 1, x: 0, y: 0 });
  const panDragRef = useRef({
    isDragging: false,
    startX: 0,
    startY: 0,
    startViewX: 0,
    startViewY: 0,
  });
  const touchRef = useRef({
    lastTouchX: 0,
    lastTouchY: 0,
    lastDist: 0,
    isTwoFinger: false,
    startX: 0,
    startY: 0,
    startTime: 0,
    startViewX: 0,
    startViewY: 0,
    moved: false,
  });
  const [swipeHint, setSwipeHint] = useState<"left" | "right" | null>(null);

  useEffect(() => {
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImage(img);
      onImageLoad({ width: img.width, height: img.height });
      if (wrapperRef.current) {
        const rect = wrapperRef.current.getBoundingClientRect();
        const scale =
          Math.min(rect.width / img.width, rect.height / img.height) * 0.95;
        setViewTransform({
          zoom: scale,
          x: (rect.width - img.width * scale) / 2,
          y: (rect.height - img.height * scale) / 2,
        });
      }
    };
    img.src = getWallPhotoUrl(wallId);
  }, [wallId, onImageLoad]);

  const toPixelCoords = useCallback(
    (hold: HoldDetail) => ({
      x: (hold.x / wallDimensions.width) * imageDimensions.width,
      y: (1 - hold.y / wallDimensions.height) * imageDimensions.height,
    }),
    [imageDimensions, wallDimensions],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = imageDimensions;
    canvas.width = width || 800;
    canvas.height = height || 600;
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    const startHolds = new Set(selectedHoldset?.start || []);
    const finishHolds = new Set(selectedHoldset?.finish || []);
    const handHolds = new Set(selectedHoldset?.hand || []);
    const footHolds = new Set(selectedHoldset?.foot || []);
    const usedHolds = new Set([
      ...startHolds,
      ...finishHolds,
      ...handHolds,
      ...footHolds,
    ]);
    const {
      scale: userScale,
      colorMode,
      uniformColor,
      opacity: userOpacity,
      filled,
    } = displaySettings;
    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const baseScale = height / 500;
      const radius = 10 * baseScale * userScale;
      const isUsed = usedHolds.has(hold.hold_index);
      const isStart = startHolds.has(hold.hold_index),
        isFinish = finishHolds.has(hold.hold_index);
      const isHand = handHolds.has(hold.hold_index),
        isFoot = footHolds.has(hold.hold_index);
      const baseAlpha = selectedHoldset ? (isUsed ? 1 : 0.2) : 0.5;
      const alpha = isUsed ? baseAlpha * userOpacity : baseAlpha;
      let strokeColor = HOLD_STROKE_COLOR;
      if (selectedHoldset && isUsed) {
        if (colorMode === "uniform") strokeColor = uniformColor;
        else if (isStart) strokeColor = displaySettings.categoryColors.start;
        else if (isFinish) strokeColor = displaySettings.categoryColors.finish;
        else if (isHand) strokeColor = displaySettings.categoryColors.hand;
        else if (isFoot) strokeColor = displaySettings.categoryColors.foot;
      }
      const footScale = isFoot ? 0.5 : 1;
      ctx.beginPath();
      ctx.arc(x, y, radius * footScale, 0, 2 * Math.PI);
      ctx.strokeStyle = strokeColor;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = isUsed && selectedHoldset ? baseScale * 2 : 2;
      if (selectedHoldset && isUsed && filled) {
        ctx.fillStyle = strokeColor;
        ctx.fill();
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }, [
    image,
    imageDimensions,
    holds,
    selectedHoldset,
    toPixelCoords,
    displaySettings,
  ]);

  const getImageCoords = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) * (imageDimensions.width / rect.width),
        y: (e.clientY - rect.top) * (imageDimensions.height / rect.height),
      };
    },
    [imageDimensions],
  );

  const findHoldAt = useCallback(
    (pixelX: number, pixelY: number) => {
      for (const hold of holds) {
        const { x, y } = toPixelCoords(hold);
        if (Math.sqrt((x - pixelX) ** 2 + (y - pixelY) ** 2) < 25) return hold;
      }
      return null;
    },
    [holds, toPixelCoords],
  );

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (!editing) return;
      const { x, y } = getImageCoords(e);
      const hold = findHoldAt(x, y);
      if (hold) onHoldClick(hold.hold_index);
    },
    [editing, getImageCoords, findHoldAt, onHoldClick],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      panDragRef.current = {
        isDragging: false,
        startX: e.clientX,
        startY: e.clientY,
        startViewX: viewTransform.x,
        startViewY: viewTransform.y,
      };
    },
    [viewTransform],
  );

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (e.buttons !== 1) return;
    const dx = e.clientX - panDragRef.current.startX;
    const dy = e.clientY - panDragRef.current.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3)
      panDragRef.current.isDragging = true;
    if (panDragRef.current.isDragging)
      setViewTransform((prev) => ({
        ...prev,
        x: panDragRef.current.startViewX + dx,
        y: panDragRef.current.startViewY + dy,
      }));
  }, []);

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!panDragRef.current.isDragging) handleCanvasClick(e);
      panDragRef.current.isDragging = false;
    },
    [handleCanvasClick],
  );

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const rect = element.getBoundingClientRect();
      const mouseX = e.clientX - rect.left,
        mouseY = e.clientY - rect.top;
      setViewTransform((prev) => {
        const newZoom = Math.max(0.1, Math.min(10, prev.zoom * zoomFactor));
        const scale = newZoom / prev.zoom;
        return {
          zoom: newZoom,
          x: mouseX - (mouseX - prev.x) * scale,
          y: mouseY - (mouseY - prev.y) * scale,
        };
      });
    };
    element.addEventListener("wheel", handleWheel, { passive: false });
    return () => element.removeEventListener("wheel", handleWheel);
  }, []);

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;
    const getTouchDist = (t: TouchList) =>
      Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const getTouchMid = (t: TouchList) => ({
      x: (t[0].clientX + t[1].clientX) / 2,
      y: (t[0].clientY + t[1].clientY) / 2,
    });
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        touchRef.current = {
          ...touchRef.current,
          lastTouchX: e.touches[0].clientX,
          lastTouchY: e.touches[0].clientY,
          startX: e.touches[0].clientX,
          startY: e.touches[0].clientY,
          startTime: Date.now(),
          isTwoFinger: false,
          moved: false,
        };
        setViewTransform((prev) => {
          touchRef.current.startViewX = prev.x;
          touchRef.current.startViewY = prev.y;
          return prev;
        });
      } else if (e.touches.length === 2) {
        e.preventDefault();
        touchRef.current.isTwoFinger = true;
        touchRef.current.lastDist = getTouchDist(e.touches);
        const mid = getTouchMid(e.touches);
        touchRef.current.lastTouchX = mid.x;
        touchRef.current.lastTouchY = mid.y;
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        e.preventDefault();
        const rect = element.getBoundingClientRect();
        const mid = getTouchMid(e.touches);
        const dist = getTouchDist(e.touches);
        const pinchFactor = dist / (touchRef.current.lastDist || dist);
        const dx = mid.x - touchRef.current.lastTouchX,
          dy = mid.y - touchRef.current.lastTouchY;
        const midX = mid.x - rect.left,
          midY = mid.y - rect.top;
        setViewTransform((prev) => {
          const newZoom = Math.max(0.1, Math.min(10, prev.zoom * pinchFactor));
          const scale = newZoom / prev.zoom;
          return {
            zoom: newZoom,
            x: midX - (midX - prev.x) * scale + dx,
            y: midY - (midY - prev.y) * scale + dy,
          };
        });
        touchRef.current.lastDist = dist;
        touchRef.current.lastTouchX = mid.x;
        touchRef.current.lastTouchY = mid.y;
      } else if (e.touches.length === 1 && !touchRef.current.isTwoFinger) {
        e.preventDefault();
        const dx = e.touches[0].clientX - touchRef.current.startX;
        const dy = e.touches[0].clientY - touchRef.current.startY;
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) touchRef.current.moved = true;
        if (touchRef.current.moved)
          setViewTransform((prev) => ({
            ...prev,
            x: touchRef.current.startViewX + dx,
            y: touchRef.current.startViewY + dy,
          }));
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2) touchRef.current.isTwoFinger = false;
      if (e.changedTouches.length === 1 && !touchRef.current.isTwoFinger) {
        const dx = e.changedTouches[0].clientX - touchRef.current.startX;
        const dy = e.changedTouches[0].clientY - touchRef.current.startY;
        const dt = Date.now() - touchRef.current.startTime;
        if (
          Math.abs(dx) > 60 &&
          Math.abs(dy) < 80 &&
          dt < 400 &&
          Math.abs(dx) > Math.abs(dy) * 1.5
        ) {
          if (dx < 0 && onSwipeNext) {
            setSwipeHint("left");
            setTimeout(() => setSwipeHint(null), 600);
            onSwipeNext();
          } else if (dx > 0 && onSwipePrev) {
            setSwipeHint("right");
            setTimeout(() => setSwipeHint(null), 600);
            onSwipePrev();
          }
        }
      }
    };
    element.addEventListener("touchstart", onTouchStart, { passive: false });
    element.addEventListener("touchmove", onTouchMove, { passive: false });
    element.addEventListener("touchend", onTouchEnd, { passive: true });
    return () => {
      element.removeEventListener("touchstart", onTouchStart);
      element.removeEventListener("touchmove", onTouchMove);
      element.removeEventListener("touchend", onTouchEnd);
    };
  }, []);

  const { zoom, x, y } = viewTransform;
  const { width, height } = imageDimensions;

  return (
    <div
      ref={wrapperRef}
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "var(--bg)",
        position: "relative",
        cursor: editing ? "crosshair" : "grab",
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        panDragRef.current.isDragging = false;
      }}
    >
      <div style={{ transform: `translate(${x}px, ${y}px)` }}>
        <canvas
          ref={canvasRef}
          style={{
            width: (width || 800) * zoom,
            height: (height || 600) * zoom,
          }}
        />
      </div>

      {swipeHint && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
          }}
        >
          <div
            className="bz-mono"
            style={{
              background: "rgba(0,0,0,0.7)",
              borderRadius: "var(--radius)",
              padding: "10px 20px",
              color: "var(--cyan)",
              fontSize: "0.7rem",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              border: "1px solid var(--cyan)",
            }}
          >
            {swipeHint === "left" ? "→ Next" : "← Prev"}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

function GeneratePage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const { climb: climbParam } = Route.useSearch();
  const wallId = wall.metadata.id;
  const wallDimensions = {
    width: wall.metadata.dimensions[0],
    height: wall.metadata.dimensions[1],
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
  const [waking, setWaking] = useState(false);
  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>(
    DEFAULT_DISPLAY_SETTINGS,
  );
  const [showDisplaySettings, setShowDisplaySettings] = useState(false);
  const [editing, setEditing] = useState(false);
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
    setIsGenerating(true);
    setError(null);
    setWaking(false);
    setEditing(false);
    const generate_grade = grade ?? gradeOptions[0];
    const request: GenerateRequest = {
      num_climbs: numClimbs ?? 3,
      grade: generate_grade,
      grade_scale: gradingScale,
      angle: angle ?? wall.metadata.angle,
    };
    const deadline = Date.now() + 30000;
    const retry_interval = 1500;
    while (true) {
      try {
        const response = await generateClimbs(
          wallId,
          request,
          generateSettings,
        );
        const named: NamedHoldset[] = response.climbs.map((holdset) => ({
          name: generateClimbName(),
          grade: generate_grade,
          angle: request.angle?.toString() ?? "45",
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
          to: "/$wallId",
          params: { wallId },
          search: {},
          replace: true,
        });
        setWaking(false);
        setIsGenerating(false);
        return;
      } catch (err) {
        if (is502(err) && Date.now() < deadline) {
          setWaking(true);
          setIsGenerating(false);
          await new Promise<void>((resolve) =>
            setTimeout(resolve, retry_interval),
          );
        } else {
          setError(err instanceof Error ? err.message : "Generation failed");
          setIsGenerating(false);
          return;
        }
      }
    }
  }, [
    wallId,
    numClimbs,
    grade,
    gradingScale,
    gradeOptions,
    generateSettings,
    wall.metadata.angle,
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

  const handleToggleEditing = useCallback(() => {
    if (!selectedHoldset) return;
    setEditing((prev) => !prev);
  }, [selectedHoldset]);

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
      .writeText(buildShareUrl(wallId, selectedClimb))
      .then(() => {
        setLinkCopied(true);
        setTimeout(() => setLinkCopied(false), 2000);
      });
  }, [wallId, selectedClimb]);

  const handleExportImage = useCallback(async () => {
    if (!selectedClimb) return;
    setIsExporting(true);
    try {
      const blob = await renderExportImage(
        wallId,
        wall.metadata.name,
        wall.holds ?? [],
        wallDimensions,
        selectedClimb.holdset,
        selectedClimb.name,
        displaySettings,
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
  }, [wallId, wall, wallDimensions, selectedClimb, displaySettings]);

  const handleNativeShare = useCallback(async () => {
    if (!selectedClimb) return;
    try {
      const url = buildShareUrl(wallId, selectedClimb);
      let file: File | undefined;
      try {
        const blob = await renderExportImage(
          wallId,
          wall.metadata.name,
          wall.holds ?? [],
          wallDimensions,
          selectedClimb.holdset,
          selectedClimb.name,
          displaySettings,
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
    wallId,
    wall,
    wallDimensions,
    selectedClimb,
    displaySettings,
    handleCopyLink,
  ]);

  const handleHoldClick = useCallback(
    (holdIndex: number) => {
      if (selectedIndex === null) return;
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
    [selectedIndex],
  );

  const handleSelectClimb = useCallback((index: number) => {
    setSelectedIndex(index);
    setEditing(false);
  }, []);

  const handleSwipeNext = useCallback(() => {
    if (generatedClimbs.length === 0) return;
    setSelectedIndex((prev) =>
      prev === null ? 0 : (prev + 1) % generatedClimbs.length,
    );
    setEditing(false);
  }, [generatedClimbs.length]);

  const handleSwipePrev = useCallback(() => {
    if (generatedClimbs.length === 0) return;
    setSelectedIndex((prev) =>
      prev === null
        ? generatedClimbs.length - 1
        : (prev - 1 + generatedClimbs.length) % generatedClimbs.length,
    );
    setEditing(false);
  }, [generatedClimbs.length]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept arrow keys if the user is typing in an input or select
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (e.key === "ArrowLeft") {
        if (generatedClimbs.length > 1) handleSwipePrev();
      } else if (e.key === "ArrowRight") {
        if (generatedClimbs.length > 1) handleSwipeNext();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
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
    angleFixed: !!wall.metadata.angle,
    onAngleChange: setAngle,
    generateSettings,
    onGenerateSettingsChange: setGenerateSettings,
    showModelSettings,
    onToggleModelSettings: () => setShowModelSettings((v) => !v),
    isGenerating,
    waking,
    error,
    onGenerate: handleGenerate,
    holdsets: generatedClimbs,
    selectedIndex,
    onSelectHoldset: handleSelectClimb,
    onDeleteHoldset: handleDeleteClimb,
    onClearHoldsets: handleClearClimbs,
  };

  const editPanelProps = {
    editing,
    onToggleEditing: handleToggleEditing,
    onReset: handleResetHoldset,
    onExportImage: handleExportImage,
    onCopyLink: handleCopyLink,
    onNativeShare: handleNativeShare,
    isExporting,
    linkCopied,
    hasNativeShare,
    climb: selectedClimb,
    gradeOptions,
    onUpdateClimb: handleUpdateClimb,
    displaySettings,
  };

  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        select option { background: #111113; color: #f4f4f5; }
      `}</style>

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
        {/* ── Header ── */}
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
          {/* Left: back + wall name */}
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

            {/* Cyan accent + wall name */}
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
                {wall.metadata.name}
              </span>
            </div>
          </div>

          {/* Center: Feedback link */}
          <a
            href={
              "https://docs.google.com/forms/d/e/1FAIpQLSeYDIel5MMjj0X3zlXFe4N4FZdUcXadAL5bR-Wjb4W7SVZ5SQ/viewform?usp=dialog"
            }
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

          {/* Right: display settings */}
          <div
            style={{
              position: "relative",
              display: "flex",
              alignItems: "center",
              gap: "10px",
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
              {/* Left Chevron */}
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

              {/* Center Title Chip */}
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
                  style={{
                    color: "var(--text-primary)",
                  }}
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

              {/* Right Chevron */}
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

        {/* ── Body ── */}
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
              wallId={wallId}
              holds={wall.holds ?? []}
              wallDimensions={wallDimensions}
              selectedHoldset={selectedHoldset}
              imageDimensions={imageDimensions}
              onImageLoad={handleImageLoad}
              displaySettings={displaySettings}
              editing={editing}
              onHoldClick={handleHoldClick}
            />
          </div>
          {/* Right panel (desktop) */}
          <div
            style={{ width: "260px", flexShrink: 0 }}
            className="hidden lg:flex"
          >
            <EditPanel {...editPanelProps} />
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
                      Edit & Share
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
                <div style={{ flex: 1, overflowY: "auto" }}>
                  <EditPanel {...editPanelProps} />
                </div>
              </div>
            </div>
          )}
          {/* Mobile FABs */}
          {generatedClimbs.length > 1 && (
            <div
              style={{
                position: "absolute",
                bottom: "96px",
                left: 0,
                right: 0,
                justifyContent: "center",
                gap: "10px",
                zIndex: 30,
                pointerEvents: "auto",
                padding: "0 16px",
              }}
              className="flex lg:hidden"
            >
              {/* Left Chevron */}
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
              {/* Right Chevron */}
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
            </div>
          )}

          <div
            style={{
              position: "absolute",
              bottom: "48px",
              left: 0,
              right: 0,
              justifyContent: "center",
              gap: "10px",
              zIndex: 30,
              pointerEvents: "none",
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

            <button
              onClick={handleNativeShare}
              disabled={!selectedClimb}
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
                background: selectedClimb ? "var(--cyan)" : "var(--surface)",
                border: `1px solid ${selectedClimb ? "var(--cyan)" : "var(--border)"}`,
                color: selectedClimb ? "#09090b" : "var(--text-dim)",
                fontWeight: selectedClimb ? 700 : 400,
                cursor: selectedClimb ? "pointer" : "not-allowed",
                borderRadius: "var(--radius)",
                boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
              }}
            >
              <Share2 size={12} /> Share
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
