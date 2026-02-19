import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { generateClimbs } from "@/api/generate";
import {
  ArrowLeft,
  Sparkles,
  Loader2,
  Settings,
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
} from "lucide-react";
import type {
  WallDetail,
  HoldDetail,
  Holdset,
  GenerateRequest,
  GenerateSettings,
  GradeScale,
} from "@/types";
import { DEFAULT_GENERATE_SETTINGS } from "@/types";

// --- Route Definition ---

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
  staleTime: 3_600_000, // 1-hour cache on GetWall
});

// --- Hold category types (from create.tsx) ---
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

// --- Hold color constants ---
const HOLD_STROKE_COLOR = "#00b679";

// --- Display settings types ---
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

// --- V-grade options ---
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

// --- Random Climb Name Generator ---
const ADJECTIVES = [
  "Angry",
  "Bold",
  "Cosmic",
  "Daring",
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
  "Xerus",
  "Yak",
  "Zebra",
];

function generateClimbName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const animal = ANIMALS[Math.floor(Math.random() * ANIMALS.length)];
  return `${adj} ${animal}`;
}

// --- Holdset with name ---

interface NamedHoldset {
  name: string;
  holdset: Holdset;
  grade: string;
}

// ============================================================
// --- Sharing utilities ---
// ============================================================

/** Encode a named holdset into a compact base64 URL-safe string. */
function encodeClimbToParam(entry: NamedHoldset): string {
  const compact = {
    n: entry.name,
    g: entry.grade,
    s: entry.holdset.start,
    f: entry.holdset.finish,
    h: entry.holdset.hand,
    t: entry.holdset.foot,
  };
  const json = JSON.stringify(compact);
  return btoa(json).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Decode a base64 URL-safe string back to a NamedHoldset, or null on failure. */
function decodeClimbFromParam(param: string): NamedHoldset | null {
  try {
    let b64 = param.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4 !== 0) b64 += "=";
    const json = atob(b64);
    const compact = JSON.parse(json);
    if (!compact) return null;
    return {
      name: typeof compact.n !== "string" ? compact.n : "Unnamed",
      grade: typeof compact.g === "string" ? compact.g : "V?",
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

/** Build a shareable URL for the current climb. */
function buildShareUrl(wallId: string, entry: NamedHoldset): string {
  const encoded = encodeClimbToParam(entry);
  const base = `${window.location.origin}/${wallId}`;
  return `${base}?climb=${encoded}`;
}

// ============================================================
// --- Export canvas renderer ---
// ============================================================

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

  const imgW = img.width;
  const imgH = img.height;

  const topBannerH = Math.round(imgH * 0.06);
  const bottomBannerH = Math.round(imgH * 0.045);
  const totalH = imgH + topBannerH + bottomBannerH;

  const canvas = document.createElement("canvas");
  canvas.width = imgW;
  canvas.height = totalH;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, imgW, totalH);

  ctx.fillStyle = "#18181b";
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

  const startSet = new Set(holdset.start);
  const finishSet = new Set(holdset.finish);
  const handSet = new Set(holdset.hand);
  const footSet = new Set(holdset.foot);
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
    const isStart = startSet.has(hold.hold_index);
    const isFinish = finishSet.has(hold.hold_index);
    const isHand = handSet.has(hold.hold_index);
    const isFoot = footSet.has(hold.hold_index);

    const baseAlpha = isUsed ? 1 : 0.15;
    const alpha = isUsed ? baseAlpha * userOpacity : baseAlpha;

    let color = HOLD_STROKE_COLOR;
    if (isUsed) {
      if (colorMode === "uniform") {
        color = uniformColor;
      } else {
        if (isStart) color = CATEGORY_COLORS.start;
        else if (isFinish) color = CATEGORY_COLORS.finish;
        else if (isHand) color = CATEGORY_COLORS.hand;
        else if (isFoot) color = CATEGORY_COLORS.foot;
      }
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
  ctx.fillStyle = "#18181b";
  ctx.fillRect(0, legendY, imgW, bottomBannerH);

  const legendFont = Math.round(bottomBannerH * 0.45);
  ctx.font = `${legendFont}px sans-serif`;
  ctx.textBaseline = "middle";
  const legendMidY = legendY + bottomBannerH / 2;
  const dotR = Math.round(bottomBannerH * 0.15);
  const pad = Math.round(imgW * 0.02);

  const legendItems = [
    { label: "Start", color: CATEGORY_COLORS.start },
    { label: "Finish", color: CATEGORY_COLORS.finish },
    { label: "Hand", color: CATEGORY_COLORS.hand },
    { label: "Foot", color: CATEGORY_COLORS.foot },
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

// --- Display Settings Panel ---

interface DisplaySettingsPanelProps {
  settings: DisplaySettings;
  onChange: (settings: DisplaySettings) => void;
}

function DisplaySettingsPanel({
  settings,
  onChange,
}: DisplaySettingsPanelProps) {
  const update = (patch: Partial<DisplaySettings>) =>
    onChange({ ...settings, ...patch });
  const updateCategoryColor = (cat: HoldCategory, color: string) => {
    update({
      categoryColors: {
        ...settings.categoryColors,
        [cat]: color,
      },
    });
  };
  return (
    <div className="space-y-4 min-w-[240px]">
      {/* Scale */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-zinc-500 uppercase tracking-wider">
            Scale
          </label>
          <span className="text-xs text-zinc-400 font-mono">
            {settings.scale.toFixed(1)}x
          </span>
        </div>
        <input
          type="range"
          min={0.3}
          max={3.0}
          step={0.1}
          value={settings.scale}
          onChange={(e) => update({ scale: parseFloat(e.target.value) })}
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer"
        />
      </div>

      {/* Color Mode */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider block mb-1.5">
          Color
        </label>
        <div className="flex gap-1">
          <button
            onClick={() => update({ colorMode: "role" })}
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
              settings.colorMode === "role"
                ? "bg-emerald-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            By Role
          </button>
          <button
            onClick={() => update({ colorMode: "uniform" })}
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
              settings.colorMode === "uniform"
                ? "bg-emerald-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Uniform
          </button>
        </div>
        {settings.colorMode === "uniform" ? (
          <div className="mt-2 flex items-center gap-2">
            <input
              type="color"
              value={settings.uniformColor}
              onChange={(e) => update({ uniformColor: e.target.value })}
              className="w-8 h-8 rounded border border-zinc-700 bg-transparent cursor-pointer"
            />
            <span className="text-xs text-zinc-500 font-mono">
              {settings.uniformColor}
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {CATEGORY_ORDER.map((cat) => (
              <div
                key={cat}
                className="flex items-center gap-2 p-1.5 rounded bg-zinc-800/30 border border-zinc-800"
              >
                <input
                  type="color"
                  value={settings.categoryColors[cat]}
                  onChange={(e) => updateCategoryColor(cat, e.target.value)}
                  className="w-6 h-6 rounded border-0 p-0 bg-transparent cursor-pointer"
                />
                <div className="flex flex-col min-w-0">
                  <span className="text-xs text-zinc-300 font-medium truncate">
                    {CATEGORY_LABELS[cat]}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Opacity */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-zinc-500 uppercase tracking-wider">
            Opacity
          </label>
          <span className="text-xs text-zinc-400 font-mono">
            {Math.round(settings.opacity * 100)}%
          </span>
        </div>
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={settings.opacity}
          onChange={(e) => update({ opacity: parseFloat(e.target.value) })}
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer"
        />
      </div>

      {/* Fill / Outline */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider block mb-1.5">
          Style
        </label>
        <div className="flex gap-1">
          <button
            onClick={() => update({ filled: true })}
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
              settings.filled
                ? "bg-emerald-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Filled
          </button>
          <button
            onClick={() => update({ filled: false })}
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
              !settings.filled
                ? "bg-emerald-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Outline
          </button>
        </div>
      </div>
    </div>
  );
}

// --- Model Settings Panel ---

interface ModelSettingsPanelProps {
  settings: GenerateSettings;
  onChange: (settings: GenerateSettings) => void;
}

function ModelSettingsPanel({ settings, onChange }: ModelSettingsPanelProps) {
  const update = (patch: Partial<GenerateSettings>) =>
    onChange({ ...settings, ...patch });

  const isDefault =
    settings.timesteps === DEFAULT_GENERATE_SETTINGS.timesteps &&
    settings.t_start_projection ===
      DEFAULT_GENERATE_SETTINGS.t_start_projection &&
    settings.x_offset === DEFAULT_GENERATE_SETTINGS.x_offset;

  return (
    <div className="space-y-4 min-w-[240px]">
      {/* Timesteps */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-zinc-500 uppercase tracking-wider">
            Timesteps
          </label>
          <span className="text-xs text-zinc-400 font-mono">
            {settings.timesteps}
          </span>
        </div>
        <input
          type="range"
          min={5}
          max={100}
          step={5}
          value={settings.timesteps}
          onChange={(e) => update({ timesteps: parseInt(e.target.value) })}
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer"
        />
        <div className="flex justify-between mt-1">
          <span className="text-xs text-zinc-600">Faster</span>
          <span className="text-xs text-zinc-600">Higher Quality</span>
        </div>
      </div>

      {/* Projection start */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-zinc-500 uppercase tracking-wider">
            Projection Start
          </label>
          <span className="text-xs text-zinc-400 font-mono">
            t={settings.t_start_projection.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min={0.3}
          max={1.0}
          step={0.05}
          value={settings.t_start_projection}
          onChange={(e) =>
            update({ t_start_projection: parseFloat(e.target.value) })
          }
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer"
        />
        <div className="flex justify-between mt-1">
          <span className="text-xs text-zinc-600">Later (Faster)</span>
          <span className="text-xs text-zinc-600">Earlier</span>
        </div>
      </div>

      {/* X Offset */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-zinc-500 uppercase tracking-wider">
            X Offset
          </label>
          <span className="text-xs text-zinc-400 font-mono">
            {settings.x_offset != null ? settings.x_offset.toFixed(2) : "Auto"}
          </span>
        </div>
        <input
          type="range"
          min={-1.5}
          max={1.5}
          step={0.05}
          value={settings.x_offset ?? 0}
          onChange={(e) => update({ x_offset: parseFloat(e.target.value) })}
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer"
        />
        <button
          onClick={() => update({ x_offset: null })}
          className={`mt-1.5 w-full text-xs py-1 rounded transition-colors ${
            settings.x_offset == null
              ? "bg-emerald-600/20 text-emerald-400 border border-emerald-700"
              : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
          }`}
        >
          {settings.x_offset == null ? "Auto (random)" : "Reset to Auto"}
        </button>
      </div>

      {/* Deterministic Generation toggle */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider block mb-1.5">
          Generation Style
        </label>
        <div className="flex gap-1">
          <button
            onClick={() =>
              update({
                deterministic: true,
              })
            }
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
              settings.deterministic
                ? "bg-emerald-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Deterministic
          </button>
          <button
            onClick={() =>
              update({
                deterministic: false,
              })
            }
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
              settings.deterministic
                ? "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                : "bg-emerald-600 text-white"
            }`}
          >
            Nondeterministic
          </button>
        </div>
      </div>

      {/* Reset to defaults */}
      {!isDefault && (
        <button
          onClick={() => onChange({ ...DEFAULT_GENERATE_SETTINGS })}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
        >
          <RotateCcw className="w-3 h-3" />
          Reset to Defaults
        </button>
      )}

      <p className="text-xs text-zinc-600 leading-relaxed pt-1 border-t border-zinc-800">
        Lower timesteps are faster but may reduce climb variety. Projection
        start controls when holds are snapped to the wall manifold.
      </p>
    </div>
  );
}

// --- GenerationPanel Component ---

interface GenerationPanelProps {
  gradingScale: GradeScale;
  gradeOptions: string[];
  grade: string;
  onGradingScaleChange: (scale: GradeScale) => void;
  onGradeChange: (grade: string) => void;
  numClimbs: number;
  onNumClimbsChange: (n: number) => void;
  angle: number | null;
  angleFixed: boolean;
  onAngleChange: (angle: number | null) => void;
  deterministic: boolean;
  onDeterministicChange: (d: boolean) => void;
  generateSettings: GenerateSettings;
  onGenerateSettingsChange: (s: GenerateSettings) => void;
  showModelSettings: boolean;
  onToggleModelSettings: () => void;
  isGenerating: boolean;
  error: string | null;
  onGenerate: () => void;
  holdsets: NamedHoldset[];
  selectedIndex: number | null;
  onSelectHoldset: (i: number) => void;
  onDeleteHoldset: (i: number) => void;
  onClearHoldsets: () => void;
}

function GenerationPanel({
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
  holdsets,
  selectedIndex,
  onSelectHoldset,
  onDeleteHoldset,
  onClearHoldsets,
}: GenerationPanelProps) {
  const hasCustomModelSettings =
    generateSettings.timesteps !== DEFAULT_GENERATE_SETTINGS.timesteps ||
    generateSettings.t_start_projection !==
      DEFAULT_GENERATE_SETTINGS.t_start_projection ||
    generateSettings.x_offset !== DEFAULT_GENERATE_SETTINGS.x_offset;

  return (
    <>
      {/* Generation controls */}
      <div className="p-4 border-b border-zinc-800 space-y-4 flex-shrink-0">
        <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
          Parameters
        </h2>

        {/* Grading scale toggle */}
        <div>
          <label className="text-xs text-zinc-500 uppercase tracking-wider block mb-1.5">
            Grading Scale
          </label>
          <div className="flex gap-1">
            <button
              onClick={() => onGradingScaleChange("v_grade")}
              className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
                gradingScale === "v_grade"
                  ? "bg-emerald-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              V-grade
            </button>
            <button
              onClick={() => onGradingScaleChange("font")}
              className={`flex-1 px-2 py-1.5 text-xs font-medium rounded transition-colors ${
                gradingScale === "font"
                  ? "bg-emerald-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Fontainebleau
            </button>
          </div>
        </div>

        {/* Grade */}
        <div>
          <label className="text-xs text-zinc-500 block mb-1">
            Target Grade
          </label>
          <select
            value={grade}
            onChange={(e) => onGradeChange(e.target.value)}
            className="w-full bg-zinc-800 text-zinc-100 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-500"
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
          <label className="text-xs text-zinc-500 block mb-1">
            Number of Climbs (Fewer = Faster)
          </label>
          <input
            type="number"
            min={1}
            max={15}
            value={numClimbs}
            onChange={(e) =>
              onNumClimbsChange(
                Math.max(1, Math.min(15, parseInt(e.target.value) || 1)),
              )
            }
            className="w-full bg-zinc-800 text-zinc-100 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-500"
          />
        </div>

        {/* Wall angle */}
        <div>
          <label className="text-xs text-zinc-500 block mb-1">
            Wall Angle (Degrees)
          </label>
          <input
            type="number"
            min={0}
            max={90}
            disabled={angleFixed}
            value={angle ?? ""}
            onChange={(e) => {
              if (e.target.value === "") {
                onAngleChange(null);
              } else {
                const parsed = parseInt(e.target.value);
                if (!isNaN(parsed)) {
                  onAngleChange(Math.max(0, Math.min(90, parsed)));
                }
              }
            }}
            className="w-full bg-zinc-800 text-zinc-100 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-500 disabled:opacity-50"
          />
        </div>

        {/* Model Settings collapsible */}
        <div>
          <button
            onClick={onToggleModelSettings}
            className="w-full flex items-center justify-between px-3 py-2 rounded bg-zinc-800 hover:bg-zinc-700 transition-colors text-xs text-zinc-400 hover:text-zinc-200"
          >
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" />
              Model Settings
              {hasCustomModelSettings && (
                <span
                  className="ml-1 w-1.5 h-1.5 rounded-full bg-amber-400 inline-block"
                  title="Custom settings active"
                />
              )}
            </span>
            <ChevronDown
              className={`w-3.5 h-3.5 transition-transform ${showModelSettings ? "rotate-180" : ""}`}
            />
          </button>
          {showModelSettings && (
            <div className="mt-2 p-3 bg-zinc-950 border border-zinc-800 rounded">
              <ModelSettingsPanel
                settings={generateSettings}
                onChange={onGenerateSettingsChange}
              />
            </div>
          )}
        </div>

        {/* Generate button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white font-medium rounded transition-colors text-sm"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              Generate
            </>
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="text-sm text-red-400 bg-red-900/20 border border-red-800 rounded px-3 py-2">
            {error}
          </div>
        )}
      </div>

      {/* Generated holdsets list */}
      <div className="flex-1 min-h-0 bg-zinc-900">
        <HoldsetList
          holdsets={holdsets}
          selectedIndex={selectedIndex}
          onSelect={onSelectHoldset}
          onDelete={onDeleteHoldset}
          onClear={onClearHoldsets}
        />
      </div>
    </>
  );
}

// --- HoldsetList Component ---

interface HoldsetListProps {
  holdsets: NamedHoldset[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  onDelete: (index: number) => void;
  onClear: () => void;
}

function HoldsetList({
  holdsets,
  selectedIndex,
  onSelect,
  onDelete,
  onClear,
}: HoldsetListProps) {
  if (holdsets.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 p-4">
        <Sparkles className="w-12 h-12 mb-3 opacity-50" />
        <p className="text-center">No climbs generated yet.</p>
        <p className="text-sm text-zinc-600 mt-1">
          Configure parameters and hit Generate.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0 flex items-center justify-between">
        <span className="text-xs text-zinc-500 uppercase tracking-wider">
          {holdsets.length} Climb{holdsets.length !== 1 ? "s" : ""}
        </span>
        <button
          onClick={onClear}
          className="text-xs text-zinc-500 hover:text-red-400 flex items-center gap-1 transition-colors"
        >
          <RefreshCcw className="w-3 h-3" />
          Clear
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {holdsets.map((entry, i) => {
          const isSelected = selectedIndex === i;
          const { holdset } = entry;

          return (
            <div
              key={i}
              className={`w-full group flex items-stretch border-b border-zinc-800/50 transition-colors
                ${
                  isSelected
                    ? "bg-zinc-800 border-l-2 border-l-emerald-500"
                    : "hover:bg-zinc-800/50 border-l-2 border-l-transparent"
                }`}
            >
              <button
                onClick={() => onSelect(i)}
                className="flex-1 text-left px-3 py-3 flex items-center gap-3 min-w-0"
              >
                <div className="w-10 h-10 rounded-md flex items-center justify-center text-sm font-bold flex-shrink-0 bg-zinc-800 text-zinc-300">
                  #{i + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-zinc-100 truncate">
                    {entry.name}
                  </div>
                  <div className="text-xs text-zinc-500 flex items-center gap-2 mt-0.5">
                    <span className="font-semibold text-emerald-500">
                      {entry.grade}
                    </span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: CATEGORY_COLORS.start }}
                      />
                      {holdset.start.length}
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: CATEGORY_COLORS.finish }}
                      />
                      {holdset.finish.length}
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: CATEGORY_COLORS.hand }}
                      />
                      {holdset.hand.length}
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: CATEGORY_COLORS.foot }}
                      />
                      {holdset.foot.length}
                    </span>
                  </div>
                </div>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(i);
                }}
                className="px-2 text-zinc-600 hover:text-red-400 hover:bg-zinc-800/80 transition-colors flex items-center"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Edit Panel (right sidebar) ---

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
  onUpdateClimb: (updates: Partial<NamedHoldset>) => void;
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

  return (
    <div className="w-full h-full flex flex-col bg-zinc-900 border-l border-zinc-800">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
            Edit Climb
          </h2>
          <button
            onClick={onToggleEditing}
            disabled={!holdset}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              editing
                ? "bg-amber-600 text-white"
                : holdset
                  ? "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                  : "bg-zinc-800/50 text-zinc-600 cursor-not-allowed"
            }`}
          >
            <Pencil className="w-3 h-3" />
            {editing ? "Done" : "Edit Climb"}
          </button>
        </div>
      </div>

      {/* Hold breakdown */}
      {holdset && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Total */}
          <div className="text-center py-3 bg-zinc-950 rounded-lg border border-zinc-800">
            {editing ? (
              <div className="px-3 space-y-2">
                <input
                  type="text"
                  value={climb?.name || ""}
                  onChange={(e) => onUpdateClimb({ name: e.target.value })}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-center font-semibold text-zinc-100 focus:outline-none focus:border-emerald-500"
                  placeholder="Climb Name"
                />
                <select
                  value={climb?.grade || ""}
                  onChange={(e) => onUpdateClimb({ grade: e.target.value })}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-center text-zinc-300 focus:outline-none focus:border-emerald-500"
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
                <div className="text-2xl font-semibold text-zinc-100 px-2 truncate">
                  {climb?.name}
                </div>
                <div className="text-xs text-zinc-500 uppercase tracking-wider mt-1">
                  {climb?.grade}
                </div>
              </>
            )}
          </div>

          {/* Category counts */}
          <div className="space-y-2">
            {(["start", "hand", "foot", "finish"] as HoldCategory[]).map(
              (cat) => (
                <div
                  key={cat}
                  className="flex items-center gap-2 px-3 py-2 bg-zinc-800 rounded-lg"
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: CATEGORY_COLORS[cat] }}
                  />
                  <span className="text-sm text-zinc-300">
                    {CATEGORY_LABELS[cat]}
                  </span>
                  <span className="ml-auto text-sm font-medium text-zinc-100">
                    {holdCounts[cat]}
                  </span>
                </div>
              ),
            )}
          </div>

          {/* Instructions */}
          {editing && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3">
              <p className="text-xs text-zinc-400 leading-relaxed">
                Click holds on the wall to cycle through roles, or edit the name
                and grade above.
              </p>
            </div>
          )}

          {/* Reset button */}
          {editing && (
            <button
              onClick={onReset}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-medium rounded-lg transition-colors text-sm"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset to Generated
            </button>
          )}

          {/* ---- Share / Export section ---- */}
          <div className="pt-2 border-t border-zinc-800 space-y-2">
            <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
              Share
            </h3>

            {/* Copy shareable link */}
            <button
              onClick={onCopyLink}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-medium rounded-lg transition-colors text-sm"
            >
              {linkCopied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Link Copied!</span>
                </>
              ) : (
                <>
                  <Link className="w-3.5 h-3.5" />
                  Copy Link
                </>
              )}
            </button>

            {/* Export as image */}
            <button
              onClick={onExportImage}
              disabled={isExporting}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 font-medium rounded-lg transition-colors text-sm"
            >
              {isExporting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Rendering...
                </>
              ) : (
                <>
                  <Image className="w-3.5 h-3.5" />
                  Save Image
                </>
              )}
            </button>

            {/* Native share (desktop only? or fallback) */}
            {hasNativeShare && (
              <button
                onClick={onNativeShare}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-emerald-700 hover:bg-emerald-600 text-white font-medium rounded-lg transition-colors text-sm"
              >
                <Share2 className="w-3.5 h-3.5" />
                Share...
              </button>
            )}
          </div>
        </div>
      )}

      {/* Legend (always visible at bottom) */}
      <div className="p-3 border-t border-zinc-800 flex-shrink-0 mt-auto">
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-1.5">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CATEGORY_COLORS.start }}
            />
            <span className="text-zinc-400">Start</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CATEGORY_COLORS.finish }}
            />
            <span className="text-zinc-400">Finish</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CATEGORY_COLORS.hand }}
            />
            <span className="text-zinc-400">Hand</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CATEGORY_COLORS.foot }}
            />
            <span className="text-zinc-400">Foot</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- WallCanvas Component ---

interface WallCanvasProps {
  wallId: string;
  holds: HoldDetail[];
  wallDimensions: { width: number; height: number };
  selectedHoldset: Holdset | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dimensions: { width: number; height: number }) => void;
  displaySettings: DisplaySettings;
  editing: boolean;
  onHoldClick: (holdIndex: number) => void;
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

  // Load image
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

  // Convert feet to pixel coordinates
  const toPixelCoords = useCallback(
    (hold: HoldDetail): { x: number; y: number } => {
      const { width: imgW, height: imgH } = imageDimensions;
      const { width: wallW, height: wallH } = wallDimensions;
      return {
        x: (hold.x / wallW) * imgW,
        y: (1 - hold.y / wallH) * imgH,
      };
    },
    [imageDimensions, wallDimensions],
  );

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = imageDimensions;
    canvas.width = width || 800;
    canvas.height = height || 600;

    // Clear
    ctx.fillStyle = "#18181b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(image, 0, 0);

    // Build hold sets from selected holdset
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

    // Draw all holds
    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const baseScale = height / 500;
      const radius = 10 * baseScale * userScale;

      const isUsed = usedHolds.has(hold.hold_index);
      const isStart = startHolds.has(hold.hold_index);
      const isFinish = finishHolds.has(hold.hold_index);
      const isHand = handHolds.has(hold.hold_index);
      const isFoot = footHolds.has(hold.hold_index);

      // Dim unused holds when a holdset is selected
      const baseAlpha = selectedHoldset ? (isUsed ? 1 : 0.2) : 0.5;
      const alpha = isUsed ? baseAlpha * userOpacity : baseAlpha;

      // Determine color
      let strokeColor = HOLD_STROKE_COLOR;
      if (selectedHoldset && isUsed) {
        if (colorMode === "uniform") {
          strokeColor = uniformColor;
        } else {
          if (isStart) strokeColor = CATEGORY_COLORS.start;
          else if (isFinish) strokeColor = CATEGORY_COLORS.finish;
          else if (isHand) strokeColor = CATEGORY_COLORS.hand;
          else if (isFoot) strokeColor = CATEGORY_COLORS.foot;
        }
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

  // Get image coordinates from mouse event
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

  // Find hold at position
  const findHoldAt = useCallback(
    (pixelX: number, pixelY: number): HoldDetail | null => {
      const hitRadius = 25;
      for (const hold of holds) {
        const { x, y } = toPixelCoords(hold);
        const dist = Math.sqrt((x - pixelX) ** 2 + (y - pixelY) ** 2);
        if (dist < hitRadius) return hold;
      }
      return null;
    },
    [holds, toPixelCoords],
  );

  // Handle canvas click
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (!editing) return;
      const { x, y } = getImageCoords(e);
      const hold = findHoldAt(x, y);
      if (hold) {
        onHoldClick(hold.hold_index);
      }
    },
    [editing, getImageCoords, findHoldAt, onHoldClick],
  );

  // Pan handlers — distinguish click from drag
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
    // Only pan while primary mouse button is held
    if (e.buttons !== 1) return;

    const dx = e.clientX - panDragRef.current.startX;
    const dy = e.clientY - panDragRef.current.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      panDragRef.current.isDragging = true;
    }
    if (panDragRef.current.isDragging) {
      setViewTransform((prev) => ({
        ...prev,
        x: panDragRef.current.startViewX + dx,
        y: panDragRef.current.startViewY + dy,
      }));
    }
  }, []);

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!panDragRef.current.isDragging) {
        handleCanvasClick(e);
      }
      panDragRef.current.isDragging = false;
    },
    [handleCanvasClick],
  );

  // Scroll wheel zoom
  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const rect = element.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

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

    return () => {
      element.removeEventListener("wheel", handleWheel);
    };
  }, []);

  const { zoom, x, y } = viewTransform;
  const { width, height } = imageDimensions;

  return (
    <div
      ref={wrapperRef}
      className={`w-full h-full overflow-hidden bg-zinc-950 ${
        editing ? "cursor-crosshair" : "cursor-grab active:cursor-grabbing"
      }`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        panDragRef.current.isDragging = false;
      }}
    >
      <div
        style={{
          transform: `translate(${x}px, ${y}px)`,
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: (width || 800) * zoom,
            height: (height || 600) * zoom,
          }}
        />
      </div>
    </div>
  );
}

// --- Main Page Component ---

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

  // Generation form state
  const [gradingScale, setGradingScale] = useState<GradeScale>("v_grade");
  const [gradeOptions, setGradeOptions] = useState(VGRADE_OPTIONS);

  const [numClimbs, setNumClimbs] = useState(5);
  const [grade, setGrade] = useState<string>("V4");
  const [angle, setAngle] = useState<number | null>(null);

  // Model / performance settings
  const [generateSettings, setGenerateSettings] = useState<GenerateSettings>(
    DEFAULT_GENERATE_SETTINGS,
  );
  const [showModelSettings, setShowModelSettings] = useState(false);

  // Results state
  const [generatedClimbs, setGeneratedClimbs] = useState<NamedHoldset[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Display settings
  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>(
    DEFAULT_DISPLAY_SETTINGS,
  );
  const [showDisplaySettings, setShowDisplaySettings] = useState(false);

  // Edit mode state
  const [editing, setEditing] = useState(false);
  const originalHoldsetsRef = useRef<Holdset[]>([]);

  // Share state
  const [isExporting, setIsExporting] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);
  const hasNativeShare = typeof navigator !== "undefined" && !!navigator.share;

  // Mobile drawer state
  const [mobilePanel, setMobilePanel] = useState<"none" | "left" | "right">(
    "none",
  );
  const closeMobilePanel = useCallback(() => setMobilePanel("none"), []);
  const selectedClimb =
    selectedIndex !== null ? generatedClimbs[selectedIndex] : null;
  const selectedHoldset = selectedClimb?.holdset ?? null;

  // --- Load shared climb from URL on mount ---
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
  }, []); // Only on mount

  const handleImageLoad = useCallback(
    (dimensions: { width: number; height: number }) => {
      setImageDimensions(dimensions);
    },
    [],
  );

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setError(null);
    setEditing(false);
    const generate_grade = grade ?? gradeOptions[0];

    const request: GenerateRequest = {
      num_climbs: numClimbs,
      grade: generate_grade,
      grade_scale: gradingScale,
      angle: angle ?? wall.metadata.angle,
    };

    try {
      const response = await generateClimbs(wallId, request, generateSettings);
      const named: NamedHoldset[] = response.climbs.map((holdset) => ({
        name: generateClimbName(),
        grade: generate_grade,
        holdset,
      }));

      // Append new climbs instead of replacing
      setGeneratedClimbs((prev) => [...named, ...prev]);

      // Append originals to ref
      const newOriginals = response.climbs.map((h) => ({
        start: [...h.start],
        finish: [...h.finish],
        hand: [...h.hand],
        foot: [...h.foot],
      }));
      originalHoldsetsRef.current = [
        ...originalHoldsetsRef.current,
        ...newOriginals,
      ];

      // Select the first of the *new* climbs
      if (response.climbs.length > 0) {
        setSelectedIndex(generatedClimbs.length); // The index where the new batch starts
      }

      navigate({
        to: "/$wallId",
        params: { wallId },
        search: {},
        replace: true,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Generation failed";
      setError(message);
      console.error("Generation error:", err);
    } finally {
      setIsGenerating(false);
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
    generatedClimbs.length,
  ]);

  // Grading scale change — keeps grade/options in sync
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
  // Toggle editing mode
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

  // Reset to original generated holdset
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

  // Handle Delete Single Climb
  const handleDeleteClimb = useCallback((index: number) => {
    setGeneratedClimbs((prev) => prev.filter((_, i) => i !== index));
    // Remove from originals ref
    originalHoldsetsRef.current = originalHoldsetsRef.current.filter(
      (_, i) => i !== index,
    );

    // Adjust selected index
    setSelectedIndex((current) => {
      if (current === null) return null;
      if (current === index) return null;
      if (current > index) return current - 1;
      return current;
    });
  }, []);

  // Handle Clear All Climbs
  const handleClearClimbs = useCallback(() => {
    setGeneratedClimbs([]);
    originalHoldsetsRef.current = [];
    setSelectedIndex(null);
  }, []);

  // --- Share: Copy link ---
  const handleCopyLink = useCallback(() => {
    if (!selectedClimb) return;
    const url = buildShareUrl(wallId, selectedClimb);
    navigator.clipboard.writeText(url).then(() => {
      setLinkCopied(true);
      setTimeout(() => setLinkCopied(false), 2000);
    });
  }, [wallId, selectedClimb]);

  // --- Share: Export image ---
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
      // Trigger download
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

  // --- Share: Native share API (mobile) ---
  const handleNativeShare = useCallback(async () => {
    if (!selectedClimb) return;
    try {
      const url = buildShareUrl(wallId, selectedClimb);

      // Try generating image first
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
      } catch (e) {
        console.warn("Failed to generate image for share", e);
      }

      const shareData: ShareData = {
        title: selectedClimb.name,
        text: `Check out this climb: ${selectedClimb.name}`,
        url,
      };

      if (file && navigator.canShare?.({ files: [file] })) {
        shareData.files = [file];
      }

      await navigator.share(shareData);
    } catch (err) {
      // User cancelled share or not supported — silent
      if ((err as Error).name !== "AbortError") {
        console.error("Share failed:", err);
        // Fallback to copy link if native share fails/isn't supported
        handleCopyLink();
      }
    }
  }, [
    wallId,
    wall,
    wallDimensions,
    selectedClimb,
    displaySettings,
    handleCopyLink,
  ]);

  // Handle hold click for editing (cycle through categories like create.tsx)
  const handleHoldClick = useCallback(
    (holdIndex: number) => {
      if (selectedIndex === null) return;

      setGeneratedClimbs((prev) => {
        const entry = prev[selectedIndex];
        if (!entry) return prev;

        const holdset = entry.holdset;

        // Determine current category of this hold
        let currentCat: HoldCategory | null = null;
        if (holdset.start.includes(holdIndex)) currentCat = "start";
        else if (holdset.finish.includes(holdIndex)) currentCat = "finish";
        else if (holdset.hand.includes(holdIndex)) currentCat = "hand";
        else if (holdset.foot.includes(holdIndex)) currentCat = "foot";

        // Helper to remove from all categories
        const removeFromAll = (hs: Holdset, idx: number): Holdset => ({
          start: hs.start.filter((h) => h !== idx),
          finish: hs.finish.filter((h) => h !== idx),
          hand: hs.hand.filter((h) => h !== idx),
          foot: hs.foot.filter((h) => h !== idx),
        });

        let newHoldset: Holdset;

        if (currentCat === null) {
          // Not in climb — add as hand
          newHoldset = { ...holdset, hand: [...holdset.hand, holdIndex] };
        } else {
          const currentIndex = CATEGORY_ORDER.indexOf(currentCat);
          let nextIndex = currentIndex + 1;
          const cleaned = removeFromAll(holdset, holdIndex);

          // Try to place in next category, skipping full ones
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
            // Cycled through all — remove
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

  // When selecting a different climb, exit edit mode
  const handleSelectClimb = useCallback((index: number) => {
    setSelectedIndex(index);
    setEditing(false);
  }, []);

  return (
    <div className="h-screen flex flex-col bg-zinc-950 relative">
      {/* Keyframes for mobile drawer animations */}
      <style>{`
        @keyframes slideInLeft {
          from { transform: translateX(-100%); }
          to { transform: translateX(0); }
        }
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 flex-shrink-0 z-20 relative">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate({ to: "/" })}
            className="flex items-center gap-1 text-zinc-400 hover:text-zinc-100 transition-colors text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="hidden sm:inline">Back</span>
          </button>
          <div className="w-px h-5 bg-zinc-700 hidden sm:block" />
          <h1 className="text-lg font-medium text-zinc-100 truncate">
            {wall.metadata.name}
          </h1>
        </div>

        {/* Display Settings Toggle (Desktop & Mobile) */}
        <div className="relative flex items-center gap-3">
          {/* Divider between Board Name area and Settings Label */}
          <div className="w-px h-5 bg-zinc-800 hidden sm:block" />

          {!showDisplaySettings && (
            <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
              Display Settings
            </span>
          )}

          <button
            onClick={() => setShowDisplaySettings(!showDisplaySettings)}
            className={`p-2 rounded-md transition-colors ${
              showDisplaySettings
                ? "bg-emerald-600 text-white"
                : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800"
            }`}
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* Settings Popover */}
          {showDisplaySettings && (
            <>
              <div
                className="fixed inset-0 z-40 bg-transparent"
                onClick={() => setShowDisplaySettings(false)}
              />
              <div
                className="absolute right-0 top-full mt-2 p-4 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl z-50 w-72"
                style={{ animation: "fadeIn 0.1s ease-out" }}
              >
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
                    Display Settings
                  </span>
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

      {/* Mobile Floating Climb Info Chip (Only when a climb is selected) */}
      {selectedClimb && (
        <div className="lg:hidden absolute top-16 left-0 right-0 z-10 flex justify-center pointer-events-none px-4">
          <div
            className="bg-zinc-900/90 backdrop-blur border border-zinc-700 rounded-full py-2 px-5 shadow-lg flex flex-col items-center pointer-events-auto"
            style={{ animation: "fadeIn 0.2s ease-out" }}
          >
            <span className="text-sm font-bold text-zinc-100">
              {selectedClimb.name}
            </span>
            <span className="text-xs font-medium text-emerald-400">
              {selectedClimb.grade}
            </span>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex min-h-0 relative">
        {/* Left panel — Generation controls */}
        <div className="hidden lg:flex w-80 flex-col border-r border-zinc-800 flex-shrink-0 bg-zinc-900">
          <GenerationPanel
            gradingScale={gradingScale}
            gradeOptions={gradeOptions}
            grade={grade}
            onGradingScaleChange={handleGradingScaleChange}
            onGradeChange={setGrade}
            numClimbs={numClimbs}
            onNumClimbsChange={setNumClimbs}
            angle={angle}
            angleFixed={!!wall.metadata.angle}
            onAngleChange={setAngle}
            generateSettings={generateSettings}
            onGenerateSettingsChange={setGenerateSettings}
            showModelSettings={showModelSettings}
            onToggleModelSettings={() => setShowModelSettings((v) => !v)}
            isGenerating={isGenerating}
            error={error}
            onGenerate={handleGenerate}
            holdsets={generatedClimbs}
            selectedIndex={selectedIndex}
            onSelectHoldset={handleSelectClimb}
            onDeleteHoldset={handleDeleteClimb}
            onClearHoldsets={handleClearClimbs}
          />
        </div>
        {/* Mobile left drawer */}
        {mobilePanel === "left" && (
          <div className="lg:hidden fixed inset-0 z-40 flex">
            {/* Backdrop */}
            <div
              className="absolute inset-0 bg-black/60"
              onClick={closeMobilePanel}
            />
            {/* Drawer */}
            <div
              className="relative w-80 max-w-[85vw] h-full flex flex-col bg-zinc-900 shadow-2xl z-10"
              style={{
                animation: "slideInLeft 0.2s ease-out forwards",
              }}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
                <span className="text-sm font-semibold text-zinc-200">
                  Climbs & Generation
                </span>
                <button
                  onClick={closeMobilePanel}
                  className="text-zinc-400 hover:text-zinc-100 p-1"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                <GenerationPanel
                  gradingScale={gradingScale}
                  gradeOptions={gradeOptions}
                  grade={grade}
                  onGradingScaleChange={handleGradingScaleChange}
                  onGradeChange={setGrade}
                  numClimbs={numClimbs}
                  onNumClimbsChange={setNumClimbs}
                  angle={angle}
                  angleFixed={!!wall.metadata.angle}
                  onAngleChange={setAngle}
                  generateSettings={generateSettings}
                  onGenerateSettingsChange={setGenerateSettings}
                  showModelSettings={showModelSettings}
                  onToggleModelSettings={() => setShowModelSettings((v) => !v)}
                  isGenerating={isGenerating}
                  error={error}
                  onGenerate={handleGenerate}
                  holdsets={generatedClimbs}
                  selectedIndex={selectedIndex}
                  onSelectHoldset={(i) => {
                    handleSelectClimb(i);
                    closeMobilePanel();
                  }}
                  onDeleteHoldset={handleDeleteClimb}
                  onClearHoldsets={handleClearClimbs}
                />
              </div>
            </div>
          </div>
        )}
        {/* Center — Wall Canvas */}
        <div className="flex-1 min-w-0">
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

        {/* Right panel — Edit Panel */}
        <div className="hidden lg:flex w-72 flex-shrink-0">
          <EditPanel
            editing={editing}
            onToggleEditing={handleToggleEditing}
            onReset={handleResetHoldset}
            onExportImage={handleExportImage}
            onCopyLink={handleCopyLink}
            onNativeShare={handleNativeShare}
            isExporting={isExporting}
            linkCopied={linkCopied}
            hasNativeShare={hasNativeShare}
            climb={selectedClimb}
            gradeOptions={gradeOptions}
            onUpdateClimb={handleUpdateClimb}
          />
        </div>

        {/* Mobile right drawer */}
        {mobilePanel === "right" && (
          <div className="lg:hidden fixed inset-0 z-40 flex justify-end">
            {/* Backdrop */}
            <div
              className="absolute inset-0 bg-black/60"
              onClick={closeMobilePanel}
            />
            {/* Drawer */}
            <div
              className="relative w-80 max-w-[85vw] h-full flex flex-col bg-zinc-900 shadow-2xl z-10"
              style={{
                animation: "slideInRight 0.2s ease-out forwards",
              }}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
                <span className="text-sm font-semibold text-zinc-200">
                  Edit & Share
                </span>
                <button
                  onClick={closeMobilePanel}
                  className="text-zinc-400 hover:text-zinc-100 p-1"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto">
                <EditPanel
                  editing={editing}
                  onToggleEditing={handleToggleEditing}
                  onReset={handleResetHoldset}
                  onExportImage={handleExportImage}
                  onCopyLink={handleCopyLink}
                  onNativeShare={handleNativeShare}
                  isExporting={isExporting}
                  linkCopied={linkCopied}
                  hasNativeShare={hasNativeShare}
                  climb={selectedClimb}
                  gradeOptions={gradeOptions}
                  onUpdateClimb={handleUpdateClimb}
                />
              </div>
            </div>
          </div>
        )}

        {/* ====== Mobile floating action buttons (visible only below lg) ====== */}
        <div className="lg:hidden absolute bottom-6 left-0 right-0 flex justify-center gap-3 z-30 pointer-events-none px-4">
          <button
            onClick={() =>
              setMobilePanel((p) => (p === "left" ? "none" : "left"))
            }
            className="pointer-events-auto flex items-center gap-2 px-5 py-3 bg-zinc-800 hover:bg-zinc-700 text-white font-medium rounded-full shadow-lg shadow-black/40 text-sm border border-zinc-700 transition-colors"
          >
            <Sparkles className="w-4 h-4 text-emerald-400" />
            {generatedClimbs.length > 0 ? "Climbs" : "Generate"}
          </button>

          <button
            onClick={() =>
              setMobilePanel((p) => (p === "right" ? "none" : "right"))
            }
            className="pointer-events-auto flex items-center gap-2 px-5 py-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 font-medium rounded-full shadow-lg shadow-black/40 text-sm border border-zinc-700 transition-colors"
          >
            <Pencil className="w-4 h-4" />
            Edit
          </button>

          <button
            onClick={handleNativeShare}
            disabled={!selectedClimb}
            className={`pointer-events-auto flex items-center gap-2 px-5 py-3 font-medium rounded-full shadow-lg shadow-black/40 text-sm transition-colors
              ${
                selectedClimb
                  ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                  : "bg-zinc-800 text-zinc-500 border border-zinc-700"
              }`}
          >
            <Share2 className="w-4 h-4" />
            Share
          </button>
        </div>
      </div>
    </div>
  );
}
