import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { generateClimbs } from "@/api/generate";
import { ArrowLeft, Sparkles, Loader2 } from "lucide-react";
import type { WallDetail, HoldDetail, Holdset, GenerateRequest } from "@/types";

// --- Route Definition ---

export const Route = createFileRoute("/walls/$wallId/generate")({
  component: GeneratePage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

// --- Hold color constants (matches view.tsx) ---

const START_COLOR = "#22c55e"; // green-500
const FINISH_COLOR = "#ffea00"; // yellow
const HAND_COLOR = "#3b82f6"; // blue-500
const FOOT_COLOR = "#a855f7"; // purple-500
const HOLD_STROKE_COLOR = "#00b679";

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

// --- HoldsetList Component ---

interface HoldsetListProps {
  holdsets: Holdset[];
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
        {holdsets.map((holdset, i) => {
          const isSelected = selectedIndex === i;
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
                  <div className="font-medium text-zinc-100">Climb {i + 1}</div>
                  <div className="text-xs text-zinc-500 flex items-center gap-2 mt-0.5">
                    <span>{totalHolds} holds</span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: START_COLOR }}
                      />
                      {holdset.start.length}
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: FINISH_COLOR }}
                      />
                      {holdset.finish.length}
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: HAND_COLOR }}
                      />
                      {holdset.hand.length}
                    </span>
                    <span className="flex items-center gap-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: FOOT_COLOR }}
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

// --- WallCanvas Component ---

interface WallCanvasProps {
  wallId: string;
  holds: HoldDetail[];
  wallDimensions: { width: number; height: number };
  selectedHoldset: Holdset | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dimensions: { width: number; height: number }) => void;
}

function WallCanvas({
  wallId,
  holds,
  wallDimensions,
  selectedHoldset,
  imageDimensions,
  onImageLoad,
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

    // Draw all holds
    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const scale = height / 1000;
      const radius = 10 * scale;

      const isUsed = usedHolds.has(hold.hold_index);
      const isStart = startHolds.has(hold.hold_index);
      const isFinish = finishHolds.has(hold.hold_index);
      const isHand = handHolds.has(hold.hold_index);
      const isFoot = footHolds.has(hold.hold_index);

      // Dim holds not in the climb when a holdset is selected
      const alpha = selectedHoldset ? (isUsed ? 1 : 0.2) : 0.5;

      // Determine color based on hold type
      let strokeColor = HOLD_STROKE_COLOR;
      if (selectedHoldset && isUsed) {
        if (isStart) strokeColor = START_COLOR;
        else if (isFinish) strokeColor = FINISH_COLOR;
        else if (isHand) strokeColor = HAND_COLOR;
        else if (isFoot) strokeColor = FOOT_COLOR;
      }
      const size = isFoot ? 0.5 : 1;

      ctx.beginPath();
      ctx.arc(x, y, radius * size, 0, 2 * Math.PI);
      ctx.strokeStyle = strokeColor;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = isUsed && selectedHoldset ? scale * 2 : 2;
      if (selectedHoldset && isUsed) {
        ctx.fillStyle = strokeColor;
        ctx.fill();
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }, [image, imageDimensions, holds, selectedHoldset, toPixelCoords]);

  // Pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      panDragRef.current = {
        isDragging: true,
        startX: e.clientX,
        startY: e.clientY,
        startViewX: viewTransform.x,
        startViewY: viewTransform.y,
      };
    },
    [viewTransform],
  );

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (panDragRef.current.isDragging) {
      setViewTransform((prev) => ({
        ...prev,
        x:
          panDragRef.current.startViewX +
          (e.clientX - panDragRef.current.startX),
        y:
          panDragRef.current.startViewY +
          (e.clientY - panDragRef.current.startY),
      }));
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    panDragRef.current.isDragging = false;
  }, []);

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
      className="w-full h-full overflow-hidden bg-zinc-950 cursor-grab active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
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
  const [generatedHoldsets, setGeneratedHoldsets] = useState<Holdset[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedHoldset =
    selectedIndex !== null ? generatedHoldsets[selectedIndex] : null;

  const handleImageLoad = useCallback(
    (dimensions: { width: number; height: number }) => {
      setImageDimensions(dimensions);
    },
    [],
  );

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setError(null);

    const request: GenerateRequest = {
      num_climbs: numClimbs,
      grade,
      grade_scale: "v_grade",
      angle: angle ?? wall.metadata.angle,
      deterministic,
    };

    try {
      const response = await generateClimbs(wallId, request);
      setGeneratedHoldsets(response.climbs);
      // Auto-select first climb if results returned
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

  return (
    <div className="h-screen flex flex-col bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() =>
              navigate({ to: "/walls/$wallId", params: { wallId } })
            }
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
        {/* Left panel */}
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
            {/* Wall angle adjust if allowed */}
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

          {/* Generated holdsets list */}
          <div className="flex-1 min-h-0 bg-zinc-900">
            <HoldsetList
              holdsets={generatedHoldsets}
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
            />
          </div>

          {/* Legend */}
          {generatedHoldsets.length > 0 && (
            <div className="p-3 border-t border-zinc-800 bg-zinc-900 flex-shrink-0">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: START_COLOR }}
                  />
                  <span className="text-zinc-400">Start</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: FINISH_COLOR }}
                  />
                  <span className="text-zinc-400">Finish</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: HAND_COLOR }}
                  />
                  <span className="text-zinc-400">Hand</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: FOOT_COLOR }}
                  />
                  <span className="text-zinc-400">Foot</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right panel - Wall Canvas */}
        <div className="flex-1 min-w-0">
          <WallCanvas
            wallId={wallId}
            holds={wall.holds ?? []}
            wallDimensions={wallDimensions}
            selectedHoldset={selectedHoldset}
            imageDimensions={imageDimensions}
            onImageLoad={handleImageLoad}
          />
        </div>
      </div>
    </div>
  );
}
