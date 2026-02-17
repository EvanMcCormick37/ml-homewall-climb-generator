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
} from "lucide-react";
import type { WallDetail, HoldDetail, Holdset, GenerateRequest } from "@/types";

// --- Route Definition ---

export const Route = createFileRoute("/$wallId")({
  component: GeneratePage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
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
  opacity: number;
  filled: boolean;
}

const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  scale: 2.0,
  colorMode: "uniform",
  uniformColor: "#3b82f6",
  opacity: 0.6,
  filled: true,
};

// --- V-grade options ---

const GRADE_OPTIONS = [
  "V0",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
  "V7",
  "V8",
  "V9",
  "V10",
  "V11",
  "V12",
  "V13",
  "V14",
  "V15",
  "V16",
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
  "Keen",
  "Lunar",
  "Mighty",
  "Noble",
  "Obscure",
  "Primal",
  "Quick",
  "Radical",
  "Savage",
  "Twisted",
  "Ultra",
  "Vivid",
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

  return (
    <div className="space-y-4">
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
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                     [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-emerald-500 [&::-webkit-slider-thumb]:cursor-pointer"
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
        {settings.colorMode === "uniform" && (
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
          className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                     [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-emerald-500 [&::-webkit-slider-thumb]:cursor-pointer"
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

// --- HoldsetList Component ---

interface HoldsetListProps {
  holdsets: NamedHoldset[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

function HoldsetList({ holdsets, selectedIndex, onSelect }: HoldsetListProps) {
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
      <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
        <span className="text-xs text-zinc-500 uppercase tracking-wider">
          {holdsets.length} Generated Climb{holdsets.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {holdsets.map((entry, i) => {
          const isSelected = selectedIndex === i;
          const { holdset } = entry;
          const totalHolds = new Set([
            ...holdset.start,
            ...holdset.finish,
            ...holdset.hand,
            ...holdset.foot,
          ]).size;

          return (
            <button
              key={i}
              onClick={() => onSelect(i)}
              className={`w-full text-left px-3 py-3 border-b border-zinc-800/50 transition-colors
                ${
                  isSelected
                    ? "bg-zinc-800 border-l-2 border-l-emerald-500"
                    : "hover:bg-zinc-800/50 border-l-2 border-l-transparent"
                }`}
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-md flex items-center justify-center text-sm font-bold flex-shrink-0 bg-zinc-800 text-zinc-300">
                  #{i + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-zinc-100 truncate">
                    {entry.name}
                  </div>
                  <div className="text-xs text-zinc-500 flex items-center gap-2 mt-0.5">
                    <span>{totalHolds} holds</span>
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
              </div>
            </button>
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
  holdset: Holdset | null;
  climbName: string;
}

function EditPanel({
  editing,
  onToggleEditing,
  onReset,
  holdset,
  climbName,
}: EditPanelProps) {
  const holdCounts = useMemo(() => {
    if (!holdset) return { hand: 0, foot: 0, start: 0, finish: 0 };
    return {
      hand: holdset.hand.length,
      foot: holdset.foot.length,
      start: holdset.start.length,
      finish: holdset.finish.length,
    };
  }, [holdset]);

  const totalHolds = useMemo(() => {
    if (!holdset) return 0;
    return new Set([
      ...holdset.start,
      ...holdset.finish,
      ...holdset.hand,
      ...holdset.foot,
    ]).size;
  }, [holdset]);

  return (
    <div className="w-72 flex-shrink-0 border-l border-zinc-800 bg-zinc-900 flex flex-col">
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
            {editing ? "Editing" : "Edit Climb"}
          </button>
        </div>
        {!holdset ? (
          <p className="text-xs text-zinc-600">
            Select a generated climb to view or edit its holds.
          </p>
        ) : (
          <p className="text-sm text-zinc-300 truncate" title={climbName}>
            {climbName}
          </p>
        )}
      </div>

      {/* Hold breakdown */}
      {holdset && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Total */}
          <div className="text-center py-3 bg-zinc-950 rounded-lg border border-zinc-800">
            <div className="text-2xl font-semibold text-zinc-100">
              {totalHolds}
            </div>
            <div className="text-xs text-zinc-500 uppercase tracking-wider mt-1">
              Total Holds
            </div>
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
                Click holds on the wall to cycle through:
              </p>
              <p className="text-xs text-zinc-300 mt-1 font-medium">
                Hand → Foot → Start → Finish → Remove
              </p>
              <p className="text-xs text-zinc-500 mt-2">
                Start and Finish are limited to 2 holds each.
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
        </div>
      )}

      {/* Legend (always visible at bottom) */}
      <div className="p-3 border-t border-zinc-800 flex-shrink-0">
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
    const img = new Image();
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
      const baseScale = height / 1000;
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
  const [numClimbs, setNumClimbs] = useState(5);
  const [grade, setGrade] = useState("V4");
  const [angle, setAngle] = useState<number | null>(null);
  const [deterministic, setDeterministic] = useState(false);

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
  // Store the original generated holdsets so reset works
  const originalHoldsetsRef = useRef<Holdset[]>([]);

  const selectedClimb =
    selectedIndex !== null ? generatedClimbs[selectedIndex] : null;
  const selectedHoldset = selectedClimb?.holdset ?? null;

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

    const request: GenerateRequest = {
      num_climbs: numClimbs,
      grade,
      grade_scale: "v_grade",
      angle: angle ?? wall.metadata.angle,
      deterministic,
    };

    try {
      const response = await generateClimbs(wallId, request);
      const named: NamedHoldset[] = response.climbs.map((holdset) => ({
        name: generateClimbName(),
        holdset,
      }));
      setGeneratedClimbs(named);
      originalHoldsetsRef.current = response.climbs.map((h) => ({
        start: [...h.start],
        finish: [...h.finish],
        hand: [...h.hand],
        foot: [...h.foot],
      }));
      if (response.climbs.length > 0) {
        setSelectedIndex(0);
      } else {
        setSelectedIndex(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Generation failed";
      setError(message);
      console.error("Generation error:", err);
    } finally {
      setIsGenerating(false);
    }
  }, [wallId, numClimbs, grade, deterministic, wall.metadata.angle, angle]);

  // Toggle editing mode
  const handleToggleEditing = useCallback(() => {
    if (!selectedHoldset) return;
    setEditing((prev) => !prev);
  }, [selectedHoldset]);

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
    <div className="h-screen flex flex-col bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate({ to: "/" })}
            className="flex items-center gap-1 text-zinc-400 hover:text-zinc-100 transition-colors text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <div className="w-px h-5 bg-zinc-700" />
          <h1 className="text-lg font-medium text-zinc-100">
            {wall.metadata.name}
          </h1>
          <span className="text-sm text-zinc-500">Generate Climbs</span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Left panel — Generation controls */}
        <div className="w-80 flex flex-col border-r border-zinc-800 flex-shrink-0">
          {/* Generation controls */}
          <div className="p-4 border-b border-zinc-800 bg-zinc-900 space-y-4 flex-shrink-0">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
              Parameters
            </h2>

            {/* Grade */}
            <div>
              <label className="text-xs text-zinc-500 block mb-1">
                Target Grade
              </label>
              <select
                value={grade}
                onChange={(e) => setGrade(e.target.value)}
                className="w-full bg-zinc-800 text-zinc-100 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-500"
              >
                {GRADE_OPTIONS.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </div>

            {/* Num climbs */}
            <div>
              <label className="text-xs text-zinc-500 block mb-1">
                Number of Climbs
              </label>
              <input
                type="number"
                min={1}
                max={50}
                value={numClimbs}
                onChange={(e) =>
                  setNumClimbs(
                    Math.max(1, Math.min(50, parseInt(e.target.value) || 1)),
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
                disabled={!!wall.metadata.angle}
                value={angle ?? ""}
                onChange={(e) => {
                  if (e.target.value === "") {
                    setAngle(null);
                  } else {
                    const parsed = parseInt(e.target.value);
                    if (!isNaN(parsed)) {
                      setAngle(Math.max(0, Math.min(90, parsed)));
                    }
                  }
                }}
                className="w-full bg-zinc-800 text-zinc-100 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-500"
              />
            </div>

            {/* Deterministic toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={!deterministic}
                onChange={(e) => setDeterministic(!e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
              />
              <span className="text-sm text-zinc-300">Nondeterministic</span>
            </label>

            {/* Generate button */}
            <button
              onClick={handleGenerate}
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

          {/* Display settings toggle + panel */}
          <div className="border-b border-zinc-800 bg-zinc-900 flex-shrink-0">
            <button
              onClick={() => setShowDisplaySettings((prev) => !prev)}
              className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-zinc-800/50 transition-colors"
            >
              <span className="flex items-center gap-2 text-xs font-bold text-zinc-400 uppercase tracking-wider">
                <Settings className="w-3.5 h-3.5" />
                Display Settings
              </span>
              <span
                className={`text-zinc-500 text-xs transition-transform ${
                  showDisplaySettings ? "rotate-180" : ""
                }`}
              >
                ▼
              </span>
            </button>
            {showDisplaySettings && (
              <div className="px-4 pb-4">
                <DisplaySettingsPanel
                  settings={displaySettings}
                  onChange={setDisplaySettings}
                />
              </div>
            )}
          </div>

          {/* Generated holdsets list */}
          <div className="flex-1 min-h-0 bg-zinc-900">
            <HoldsetList
              holdsets={generatedClimbs}
              selectedIndex={selectedIndex}
              onSelect={handleSelectClimb}
            />
          </div>
        </div>

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
        <EditPanel
          editing={editing}
          onToggleEditing={handleToggleEditing}
          onReset={handleResetHoldset}
          holdset={selectedHoldset}
          climbName={selectedClimb?.name ?? ""}
        />
      </div>
    </div>
  );
}
