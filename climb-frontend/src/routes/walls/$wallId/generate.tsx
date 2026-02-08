import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { generateClimbs } from "@/api/generate";
import {
  ArrowLeft,
  Sparkles,
  Loader2,
  Hash,
  ChevronDown,
  Layers,
} from "lucide-react";
import type { WallDetail, HoldDetail } from "@/types";
import type { GeneratedClimb, GradeScale } from "@/types/generate";

// --- Route Definition ---

export const Route = createFileRoute("/walls/$wallId/generate")({
  component: GenerateClimbsPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

// --- Constants ---

const GENERATED_HOLD_COLOR = "#06b6d4"; // cyan-500
const HOLD_STROKE_COLOR = "#00b679";

// V-grade options that match the DDPM's grade lookup
const V_GRADE_OPTIONS = [
  "V0-",
  "V0",
  "V0+",
  "V1",
  "V1+",
  "V2",
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

const FONT_GRADE_OPTIONS = [
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

const NUM_CLIMBS_OPTIONS = [1, 3, 5, 10, 15, 20];

// --- GenerateForm Component ---

interface GenerateFormProps {
  wallAngle: number | null;
  onGenerate: (params: {
    grade: string;
    gradeScale: GradeScale;
    angle: number | null;
  }) => void;
  isGenerating: boolean;
}

function GenerateForm({
  wallAngle,
  onGenerate,
  isGenerating,
}: GenerateFormProps) {
  const [gradeScale, setGradeScale] = useState<GradeScale>("v_grade");
  const [grade, setGrade] = useState("V4");
  const [angleOverride, setAngleOverride] = useState<number | null>(null);
  const [isGradeOpen, setIsGradeOpen] = useState(false);

  const gradeOptions =
    gradeScale === "v_grade" ? V_GRADE_OPTIONS : FONT_GRADE_OPTIONS;

  // Reset grade when switching scales
  const handleScaleChange = (scale: GradeScale) => {
    setGradeScale(scale);
    setGrade(scale === "v_grade" ? "V4" : "6a");
  };

  const displayAngle = angleOverride ?? wallAngle;

  return (
    <div className="p-4 space-y-5">
      {/* Grade Scale Toggle */}
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-2">
          Grading System
        </label>
        <div className="flex rounded-lg overflow-hidden border border-zinc-700">
          <button
            type="button"
            onClick={() => handleScaleChange("v_grade")}
            className={`flex-1 px-3 py-2 text-sm font-medium transition-colors
              ${
                gradeScale === "v_grade"
                  ? "bg-cyan-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-750"
              }`}
          >
            V-Grade
          </button>
          <button
            type="button"
            onClick={() => handleScaleChange("font")}
            className={`flex-1 px-3 py-2 text-sm font-medium transition-colors
              ${
                gradeScale === "font"
                  ? "bg-cyan-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-750"
              }`}
          >
            Font
          </button>
        </div>
      </div>

      {/* Grade Dropdown */}
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-1">
          Target Grade
        </label>
        <div className="relative">
          <button
            type="button"
            onClick={() => setIsGradeOpen(!isGradeOpen)}
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg
                       text-zinc-100 text-left flex items-center justify-between"
          >
            {grade}
            <ChevronDown className="w-4 h-4 text-zinc-500" />
          </button>
          {isGradeOpen && (
            <div
              className="absolute top-full left-0 right-0 mt-1 bg-zinc-800 border border-zinc-700
                            rounded-lg shadow-lg max-h-48 overflow-y-auto z-10"
            >
              {gradeOptions.map((g) => (
                <button
                  key={g}
                  type="button"
                  onClick={() => {
                    setGrade(g);
                    setIsGradeOpen(false);
                  }}
                  className={`w-full px-3 py-2 text-left hover:bg-zinc-700 transition-colors
                    ${grade === g ? "bg-zinc-700 text-cyan-400" : "text-zinc-300"}`}
                >
                  {g}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Angle */}
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-1">
          Angle
        </label>
        {wallAngle !== null ? (
          <div className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-400 text-sm">
            {wallAngle}° <span className="text-zinc-600">(wall default)</span>
          </div>
        ) : (
          <input
            type="number"
            value={angleOverride ?? ""}
            onChange={(e) =>
              setAngleOverride(e.target.value ? Number(e.target.value) : null)
            }
            placeholder="45"
            min={0}
            max={90}
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg
                       text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-cyan-500"
          />
        )}
      </div>

      {/* Generate Button */}
      <button
        type="button"
        onClick={() =>
          onGenerate({
            grade,
            gradeScale,
            angle: displayAngle ?? 45,
          })
        }
        disabled={isGenerating}
        className="w-full px-4 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900 disabled:cursor-not-allowed
                   rounded-lg text-white font-medium transition-colors flex items-center justify-center gap-2"
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
    </div>
  );
}

// --- GeneratedClimbList Component ---

interface GeneratedClimbListProps {
  climbs: GeneratedClimb[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  grade: string;
}

function GeneratedClimbList({
  climbs,
  selectedIndex,
  onSelect,
  grade,
}: GeneratedClimbListProps) {
  if (climbs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 p-4">
        <Layers className="w-12 h-12 mb-3 opacity-50" />
        <p className="text-center text-sm">No generated climbs yet.</p>
        <p className="text-xs text-zinc-600 mt-1">
          Configure options above and hit Generate.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
        <span className="text-xs text-zinc-500 uppercase tracking-wider">
          {climbs.length} Generated Climb{climbs.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {climbs.map((climb, i) => {
          const isSelected = selectedIndex === i;
          return (
            <button
              key={i}
              onClick={() => onSelect(i)}
              className={`w-full text-left px-3 py-3 border-b border-zinc-800/50 transition-colors
                ${
                  isSelected
                    ? "bg-zinc-800 border-l-2 border-l-cyan-500"
                    : "hover:bg-zinc-800/50 border-l-2 border-l-transparent"
                }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-md flex items-center justify-center text-sm font-bold flex-shrink-0"
                  style={{
                    backgroundColor: "rgba(6, 182, 212, 0.15)",
                    color: GENERATED_HOLD_COLOR,
                  }}
                >
                  {grade}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-zinc-100">Climb {i + 1}</div>
                  <div className="text-xs text-zinc-500 flex items-center gap-2 mt-0.5">
                    <Hash className="w-3 h-3" />
                    <span>{climb.num_holds} holds</span>
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
  selectedClimb: GeneratedClimb | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dims: { width: number; height: number }) => void;
}

function WallCanvas({
  wallId,
  holds,
  wallDimensions,
  selectedClimb,
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

    ctx.fillStyle = "#18181b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    // Build set of generated hold indices
    const generatedHolds = new Set(selectedClimb?.holds ?? []);

    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const size = height / 1800;
      const radius = hold.is_foot ? 10 * size : 18 * size;

      const isUsed = generatedHolds.has(hold.hold_index);

      // Dim holds not in the climb when a climb is selected
      const alpha = selectedClimb ? (isUsed ? 1 : 0.15) : 0.5;

      const strokeColor =
        selectedClimb && isUsed ? GENERATED_HOLD_COLOR : HOLD_STROKE_COLOR;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = strokeColor;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = isUsed && selectedClimb ? size * 2 : 2;
      if (selectedClimb && isUsed) {
        ctx.fillStyle = strokeColor;
        ctx.fill();
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }, [image, imageDimensions, holds, selectedClimb, toPixelCoords]);

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

  const { zoom, x: vx, y: vy } = viewTransform;
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
      <div style={{ transform: `translate(${vx}px, ${vy}px)` }}>
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

function GenerateClimbsPage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const wallId = wall.metadata.id;
  const wallDimensions = {
    width: wall.metadata.dimensions[0],
    height: wall.metadata.dimensions[1],
  };
  const wallAngle = wall.metadata.angle ?? null;

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [generatedClimbs, setGeneratedClimbs] = useState<GeneratedClimb[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastGrade, setLastGrade] = useState("V4");

  const handleImageLoad = useCallback(
    (dimensions: { width: number; height: number }) => {
      setImageDimensions(dimensions);
    },
    [],
  );

  const handleGenerate = useCallback(
    async (params: {
      grade: string;
      gradeScale: GradeScale;
      angle: number | null;
    }) => {
      setIsGenerating(true);
      setError(null);

      try {
        const response = await generateClimbs(wallId, {
          num_climbs: 1,
          grade: params.grade,
          grade_scale: params.gradeScale,
          angle: params.angle,
          deterministic: false,
        });

        setGeneratedClimbs((prev) => [...prev, ...response.climbs]);
        setLastGrade(params.grade);
        setSelectedIndex(response.climbs.length > 0 ? 0 : null);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Generation failed";
        setError(message);
        setGeneratedClimbs([]);
        setSelectedIndex(null);
      } finally {
        setIsGenerating(false);
      }
    },
    [wallId],
  );

  const selectedClimb =
    selectedIndex !== null ? (generatedClimbs[selectedIndex] ?? null) : null;

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
          <span className="text-sm text-zinc-500 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5" />
            Generate Climbs
          </span>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="px-6 py-2 bg-red-900/50 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Left panel */}
        <div className="w-80 flex flex-col border-r border-zinc-800 flex-shrink-0">
          {/* Generation controls */}
          <div className="border-b border-zinc-800 bg-zinc-900 flex-shrink-0">
            <div className="px-4 pt-4 pb-0">
              <h2 className="text-lg font-semibold text-zinc-100">
                Generate Climbs
              </h2>
              <p className="text-sm text-zinc-500 mt-1">
                Configure parameters and generate
              </p>
            </div>
            <GenerateForm
              wallAngle={wallAngle}
              onGenerate={handleGenerate}
              isGenerating={isGenerating}
            />
          </div>

          {/* Generated climb list */}
          <div className="flex-1 min-h-0 bg-zinc-900">
            <GeneratedClimbList
              climbs={generatedClimbs}
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
              grade={lastGrade}
            />
          </div>
        </div>

        {/* Right panel - Wall Canvas */}
        <div className="flex-1 min-w-0">
          <WallCanvas
            wallId={wallId}
            holds={wall.holds ?? []}
            wallDimensions={wallDimensions}
            selectedClimb={selectedClimb}
            imageDimensions={imageDimensions}
            onImageLoad={handleImageLoad}
          />
        </div>
      </div>
    </div>
  );
}
