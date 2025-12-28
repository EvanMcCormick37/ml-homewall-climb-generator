import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { useClimbs } from "@/hooks/useClimbs";
import { ArrowLeft, Calendar, User, Tag, Hash, Layers } from "lucide-react";
import type { Climb, WallDetail, HoldDetail, Holdset } from "@/types";
import { gradeToString, gradeToColor } from "@/utils/climbs";

// --- Route Definition ---

export const Route = createFileRoute("/walls/$wallId/view")({
  component: WallViewPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

// --- Constants for rendering ---

const START_COLOR = "#22c55e"; // green-500
const FINISH_COLOR = "#ffea00"; // yellow
const HAND_COLOR = "#3b82f6"; // blue-500
const FOOT_COLOR = "#a855f7"; // purple-500
const HOLD_STROKE_COLOR = "#00b679";

// --- ClimbList Component ---

interface ClimbListProps {
  climbs: Climb[];
  loading: boolean;
  selectedClimb: Climb | null;
  onSelectClimb: (climb: Climb) => void;
  total: number;
}

function ClimbList({
  climbs,
  loading,
  selectedClimb,
  onSelectClimb,
  total,
}: ClimbListProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse text-zinc-500">Loading climbs...</div>
      </div>
    );
  }

  if (climbs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 p-4">
        <Layers className="w-12 h-12 mb-3 opacity-50" />
        <p className="text-center">No climbs on this wall yet.</p>
        <p className="text-sm text-zinc-600 mt-1">
          Create climbs to see them here.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
        <span className="text-xs text-zinc-500 uppercase tracking-wider">
          {total} Climb{total !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {climbs.map((climb) => {
          const isSelected = selectedClimb?.id === climb.id;
          return (
            <button
              key={climb.id}
              onClick={() => onSelectClimb(climb)}
              className={`w-full text-left px-3 py-3 border-b border-zinc-800/50 transition-colors
                ${
                  isSelected
                    ? "bg-zinc-800 border-l-2 border-l-emerald-500"
                    : "hover:bg-zinc-800/50 border-l-2 border-l-transparent"
                }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-md flex items-center justify-center text-sm font-bold flex-shrink-0"
                  style={{
                    backgroundColor: `${gradeToColor(climb.grade)}20`,
                    color: gradeToColor(climb.grade),
                  }}
                >
                  {gradeToString(climb.grade)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-zinc-100 truncate">
                    {climb.name || "Unnamed"}
                  </div>
                  <div className="text-xs text-zinc-500 flex items-center gap-2 mt-0.5">
                    <span>{climb.ascents} ascents</span>
                    {climb.setter_name && (
                      <>
                        <span>•</span>
                        <span>{climb.setter_name}</span>
                      </>
                    )}
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

// --- ClimbDetails Component ---

interface ClimbDetailsProps {
  climb: Climb;
}

function ClimbDetails({ climb }: ClimbDetailsProps) {
  const createdDate = new Date(climb.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  // Get hold counts from the new Holdset structure
  const { start, finish, hand, foot } = climb.holdset;
  const totalHolds = new Set([...start, ...finish, ...hand, ...foot]).size;

  return (
    <div className="p-4 h-full overflow-y-auto">
      {/* Header with name and grade */}
      <div className="flex items-start gap-3 mb-4">
        <div
          className="w-14 h-14 rounded-lg flex items-center justify-center text-lg font-bold flex-shrink-0"
          style={{
            backgroundColor: `${gradeToColor(climb.grade)}25`,
            color: gradeToColor(climb.grade),
            border: `2px solid ${gradeToColor(climb.grade)}50`,
          }}
        >
          {gradeToString(climb.grade)}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-zinc-100 truncate">
            {climb.name || "Unnamed Climb"}
          </h3>
          <p className="text-sm text-zinc-400">{climb.ascents} ascents</p>
        </div>
      </div>

      {/* Details grid */}
      <div className="space-y-3">
        {/* Setter */}
        {climb.setter_name && (
          <div className="flex items-center gap-2 text-sm">
            <User className="w-4 h-4 text-zinc-500" />
            <span className="text-zinc-400">Setter:</span>
            <span className="text-zinc-200">{climb.setter_name}</span>
          </div>
        )}

        {/* Date */}
        <div className="flex items-center gap-2 text-sm">
          <Calendar className="w-4 h-4 text-zinc-500" />
          <span className="text-zinc-400">Created:</span>
          <span className="text-zinc-200">{createdDate}</span>
        </div>

        {/* Hold counts */}
        <div className="flex items-center gap-2 text-sm">
          <Hash className="w-4 h-4 text-zinc-500" />
          <span className="text-zinc-400">Holds:</span>
          <span className="text-zinc-200">{totalHolds} total</span>
        </div>

        {/* Tags */}
        {climb.tags && climb.tags.length > 0 && (
          <div className="flex items-start gap-2 text-sm">
            <Tag className="w-4 h-4 text-zinc-500 mt-0.5" />
            <div className="flex flex-wrap gap-1">
              {climb.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 bg-zinc-800 text-zinc-300 rounded text-xs"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// --- WallCanvas Component ---

interface WallCanvasProps {
  wallId: string;
  holds: HoldDetail[];
  wallDimensions: { width: number; height: number };
  selectedClimb: Climb | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dimensions: { width: number; height: number }) => void;
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

  // Pan drag state
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

      // Fit to container
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
    [imageDimensions, wallDimensions]
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

    // Build hold sets from selected climb
    const startHolds = new Set(selectedClimb?.holdset.start || []);
    const finishHolds = new Set(selectedClimb?.holdset.finish || []);
    const handHolds = new Set(selectedClimb?.holdset.hand || []);
    const footHolds = new Set(selectedClimb?.holdset.foot || []);
    const usedHolds = new Set([
      ...startHolds,
      ...finishHolds,
      ...handHolds,
      ...footHolds,
    ]);

    // Draw all holds
    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const radius = 15;

      const isUsed = usedHolds.has(hold.hold_index);
      const isStart = startHolds.has(hold.hold_index);
      const isFinish = finishHolds.has(hold.hold_index);
      const isHand = handHolds.has(hold.hold_index);
      const isFoot = footHolds.has(hold.hold_index);

      // Dim holds not in the climb when a climb is selected
      const alpha = selectedClimb ? (isUsed ? 1 : 0.2) : 0.5;

      // Determine color based on hold type
      let strokeColor = HOLD_STROKE_COLOR;
      if (selectedClimb && isUsed) {
        if (isStart) strokeColor = START_COLOR;
        else if (isFinish) strokeColor = FINISH_COLOR;
        else if (isHand) strokeColor = HAND_COLOR;
        else if (isFoot) strokeColor = FOOT_COLOR;
      }

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = strokeColor;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = isUsed && selectedClimb ? 4 : 2;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw hold index
      if (!selectedClimb || isUsed) {
        ctx.fillStyle = "white";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.globalAlpha = alpha;
        ctx.fillText(hold.hold_index.toString(), x, y);
        ctx.globalAlpha = 1;
      }
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
    [viewTransform]
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

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setViewTransform((prev) => ({
      ...prev,
      zoom: Math.max(0.1, Math.min(10, prev.zoom * zoomFactor)),
    }));
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
      onWheel={handleWheel}
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

function WallViewPage() {
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

  const { climbs, loading, total, selectedClimb, setSelectedClimb } = useClimbs(
    wallId,
    { limit: 100, sort_by: "date", descending: true }
  );

  const handleImageLoad = useCallback(
    (dimensions: { width: number; height: number }) => {
      setImageDimensions(dimensions);
    },
    []
  );

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
          <span className="text-sm text-zinc-500">View Climbs</span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Left panel - ClimbDetails (top) and ClimbList (bottom) */}
        <div className="w-80 flex flex-col border-r border-zinc-800 flex-shrink-0">
          {/* ClimbDetails - only show when a climb is selected */}
          {selectedClimb && (
            <div className="h-[380px] border-b border-zinc-800 bg-zinc-900 flex-shrink-0">
              <ClimbDetails climb={selectedClimb} />
            </div>
          )}

          {/* ClimbList - fills remaining space */}
          <div className="flex-1 min-h-0 bg-zinc-900">
            <ClimbList
              climbs={climbs}
              loading={loading}
              selectedClimb={selectedClimb}
              onSelectClimb={setSelectedClimb}
              total={total}
            />
          </div>
        </div>

        {/* Right panel - Wall Canvas */}
        <div className="flex-1 min-w-0">
          <WallCanvas
            wallId={wallId}
            holds={wall.holds}
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
