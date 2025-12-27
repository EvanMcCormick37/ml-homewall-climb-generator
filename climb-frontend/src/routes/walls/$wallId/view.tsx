import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { useClimbs } from "@/hooks/useClimbs";
import { ArrowLeft, Calendar, User, Tag, Hash, Layers } from "lucide-react";
import type { Climb, WallDetail, HoldDetail, HoldWithPixels } from "@/types";
import { gradeToString, gradeToColor } from "@/types/climb";

// --- Route Definition ---

export const Route = createFileRoute("/walls/$wallId/view")({
  component: WallViewPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

// --- Constants for rendering ---

const LH_COLOR = "#3b82f6"; // blue-500
const RH_COLOR = "#22c55e"; // green-500
const START_COLOR = "#facc15"; // yellow-400
const END_COLOR = "#ef4444"; // red-500
const HOLD_STROKE_COLOR = "#00b679";
const SEQUENCE_LINE_COLOR = "rgba(255, 255, 255, 0.4)";

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
                    <span>{climb.num_moves} moves</span>
                    {climb.setter && (
                      <>
                        <span>•</span>
                        <span>{climb.setter}</span>
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
  holds: HoldDetail[];
}

function ClimbDetails({ climb, holds }: ClimbDetailsProps) {
  const createdDate = new Date(climb.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  // Get all unique hold IDs used in the climb
  const usedHoldIds = new Set(climb.sequence.flat().filter((h) => h >= 0));

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
          <p className="text-sm text-zinc-400">{climb.num_moves} moves</p>
        </div>
      </div>

      {/* Details grid */}
      <div className="space-y-3">
        {/* Setter */}
        {climb.setter && (
          <div className="flex items-center gap-2 text-sm">
            <User className="w-4 h-4 text-zinc-500" />
            <span className="text-zinc-400">Setter:</span>
            <span className="text-zinc-200">{climb.setter}</span>
          </div>
        )}

        {/* Date */}
        <div className="flex items-center gap-2 text-sm">
          <Calendar className="w-4 h-4 text-zinc-500" />
          <span className="text-zinc-400">Created:</span>
          <span className="text-zinc-200">{createdDate}</span>
        </div>

        {/* Tags */}
        {climb.tags && climb.tags.length > 0 && (
          <div className="flex items-start gap-2 text-sm">
            <Tag className="w-4 h-4 text-zinc-500 mt-0.5" />
            <span className="text-zinc-400">Tags:</span>
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

        {/* Holds used */}
        <div className="flex items-start gap-2 text-sm">
          <Hash className="w-4 h-4 text-zinc-500 mt-0.5" />
          <span className="text-zinc-400">Holds:</span>
          <span className="text-zinc-200">{usedHoldIds.size} unique holds</span>
        </div>
      </div>

      {/* Sequence visualization */}
      <div className="mt-4 pt-4 border-t border-zinc-800">
        <h4 className="text-xs text-zinc-500 uppercase tracking-wider mb-2">
          Sequence
        </h4>
        <div className="flex flex-wrap gap-1">
          {climb.sequence.map((pos, idx) => {
            const isStart = idx === 0;
            const isEnd = idx === climb.sequence.length - 1;
            return (
              <div
                key={idx}
                className={`px-2 py-1 rounded text-xs font-mono flex items-center gap-1
                  ${isStart ? "bg-yellow-900/30 text-yellow-400" : ""}
                  ${isEnd ? "bg-red-900/30 text-red-400" : ""}
                  ${!isStart && !isEnd ? "bg-zinc-800 text-zinc-300" : ""}
                `}
              >
                <span className="text-blue-400">{pos[0] >= 0 ? pos[0] : "—"}</span>
                <span className="text-zinc-600">/</span>
                <span className="text-green-400">{pos[1] >= 0 ? pos[1] : "—"}</span>
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-4 mt-2 text-xs text-zinc-500">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
            <span>LH</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span>RH</span>
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
  selectedClimb: Climb | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dimensions: { width: number; height: number }) => void;
}

function WallCanvas({
  wallId,
  holds,
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
        const scale = Math.min(rect.width / img.width, rect.height / img.height) * 0.95;
        setViewTransform({
          zoom: scale,
          x: (rect.width - img.width * scale) / 2,
          y: (rect.height - img.height * scale) / 2,
        });
      }
    };
    img.src = getWallPhotoUrl(wallId);
  }, [wallId, onImageLoad]);

  // Convert normalized hold coords to pixel coords
  const getHoldPixelCoords = useCallback(
    (hold: HoldDetail): { x: number; y: number } => {
      return {
        x: hold.norm_x * imageDimensions.width,
        y: (1 - hold.norm_y) * imageDimensions.height,
      };
    },
    [imageDimensions]
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

    // Determine which holds are used in the selected climb
    const usedHoldIds = new Set<number>();
    const startHoldIds = new Set<number>();
    const endHoldIds = new Set<number>();

    if (selectedClimb) {
      selectedClimb.sequence.forEach((pos, idx) => {
        pos.forEach((holdId) => {
          if (holdId >= 0) {
            usedHoldIds.add(holdId);
            if (idx === 0) startHoldIds.add(holdId);
            if (idx === selectedClimb.sequence.length - 1) endHoldIds.add(holdId);
          }
        });
      });
    }

    // Draw all holds (dimmed if not in climb, or if no climb selected)
    holds.forEach((hold) => {
      const { x, y } = getHoldPixelCoords(hold);
      const radius = 15;

      const isUsed = usedHoldIds.has(hold.hold_id);
      const isStart = startHoldIds.has(hold.hold_id);
      const isEnd = endHoldIds.has(hold.hold_id);

      // Dim holds not in the climb when a climb is selected
      const alpha = selectedClimb ? (isUsed ? 1 : 0.2) : 0.5;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle =
        isStart && selectedClimb
          ? START_COLOR
          : isEnd && selectedClimb
          ? END_COLOR
          : HOLD_STROKE_COLOR;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = isUsed && selectedClimb ? 4 : 2;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw hold ID
      if (!selectedClimb || isUsed) {
        ctx.fillStyle = "white";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.globalAlpha = alpha;
        ctx.fillText(hold.hold_id.toString(), x, y);
        ctx.globalAlpha = 1;
      }
    });

    // Draw climb sequence if selected
    if (selectedClimb && selectedClimb.sequence.length > 0) {
      const holdMap = new Map<number, HoldDetail>();
      holds.forEach((h) => holdMap.set(h.hold_id, h));

      // Draw sequence lines and markers
      const sequence = selectedClimb.sequence;

      // Draw connection lines between consecutive positions
      for (let i = 1; i < sequence.length; i++) {
        const prevPos = sequence[i - 1];
        const currPos = sequence[i];

        // Draw lines for each limb that moved
        [0, 1].forEach((limbIdx) => {
          const prevHoldId = prevPos[limbIdx];
          const currHoldId = currPos[limbIdx];

          if (prevHoldId >= 0 && currHoldId >= 0 && prevHoldId !== currHoldId) {
            const prevHold = holdMap.get(prevHoldId);
            const currHold = holdMap.get(currHoldId);

            if (prevHold && currHold) {
              const p1 = getHoldPixelCoords(prevHold);
              const p2 = getHoldPixelCoords(currHold);

              ctx.beginPath();
              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(p2.x, p2.y);
              ctx.strokeStyle = limbIdx === 0 ? LH_COLOR : RH_COLOR;
              ctx.globalAlpha = 0.6;
              ctx.lineWidth = 2;
              ctx.setLineDash([5, 5]);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.globalAlpha = 1;
            }
          }
        });
      }

      // Draw position markers
      sequence.forEach((pos, moveIdx) => {
        const isStart = moveIdx === 0;
        const isEnd = moveIdx === sequence.length - 1;

        pos.forEach((holdId, limbIdx) => {
          if (holdId < 0) return;

          const hold = holdMap.get(holdId);
          if (!hold) return;

          const { x, y } = getHoldPixelCoords(hold);
          const offset = limbIdx === 0 ? -12 : 12; // LH left, RH right
          const markerX = x + offset;
          const markerY = y;

          // Draw marker circle
          ctx.beginPath();
          ctx.arc(markerX, markerY, 10, 0, 2 * Math.PI);
          ctx.fillStyle = isStart
            ? START_COLOR
            : isEnd
            ? END_COLOR
            : limbIdx === 0
            ? LH_COLOR
            : RH_COLOR;
          ctx.fill();
          ctx.strokeStyle = "white";
          ctx.lineWidth = 2;
          ctx.stroke();

          // Draw move number
          ctx.fillStyle = isStart || isEnd ? "#000" : "white";
          ctx.font = "bold 9px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText((moveIdx + 1).toString(), markerX, markerY);
        });
      });

      // Draw legend
      const legendX = 15;
      const legendY = 15;

      ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
      ctx.fillRect(legendX, legendY, 140, 90);
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, legendY, 140, 90);

      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";

      // LH legend
      ctx.fillStyle = LH_COLOR;
      ctx.beginPath();
      ctx.arc(legendX + 15, legendY + 22, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.fillText("Left Hand", legendX + 30, legendY + 22);

      // RH legend
      ctx.fillStyle = RH_COLOR;
      ctx.beginPath();
      ctx.arc(legendX + 15, legendY + 44, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.fillText("Right Hand", legendX + 30, legendY + 44);

      // Move count
      ctx.fillStyle = "#aaa";
      ctx.font = "11px sans-serif";
      ctx.fillText(
        `${selectedClimb.sequence.length} moves`,
        legendX + 10,
        legendY + 70
      );
      ctx.fillText(
        `${gradeToString(selectedClimb.grade)}`,
        legendX + 80,
        legendY + 70
      );
    }
  }, [image, imageDimensions, holds, selectedClimb, getHoldPixelCoords]);

  // Handle wheel zoom
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const rect = wrapperRef.current?.getBoundingClientRect();
      if (!rect) return;

      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      setViewTransform((prev) => {
        const newZoom = Math.max(0.1, Math.min(5, prev.zoom * delta));
        const zoomRatio = newZoom / prev.zoom;
        return {
          zoom: newZoom,
          x: mouseX - (mouseX - prev.x) * zoomRatio,
          y: mouseY - (mouseY - prev.y) * zoomRatio,
        };
      });
    },
    []
  );

  // Handle mouse down for panning
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

  // Handle mouse move for panning
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!panDragRef.current.isDragging) return;

    const dx = e.clientX - panDragRef.current.startX;
    const dy = e.clientY - panDragRef.current.startY;

    setViewTransform({
      zoom: viewTransform.zoom,
      x: panDragRef.current.startViewX + dx,
      y: panDragRef.current.startViewY + dy,
    });
  }, [viewTransform.zoom]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    panDragRef.current.isDragging = false;
  }, []);

  // Setup wheel listener
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    wrapper.addEventListener("wheel", handleWheel, { passive: false });
    return () => wrapper.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

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

function WallViewPage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const wallId = wall.metadata.id;

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });

  const {
    climbs,
    loading,
    total,
    selectedClimb,
    setSelectedClimb,
  } = useClimbs(wallId, { limit: 100, sort_by: "date", descending: true });

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
            onClick={() => navigate({ to: "/walls/$wallId", params: { wallId } })}
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
            <div className="h-[320px] border-b border-zinc-800 bg-zinc-900 flex-shrink-0">
              <ClimbDetails climb={selectedClimb} holds={wall.holds} />
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
            selectedClimb={selectedClimb}
            imageDimensions={imageDimensions}
            onImageLoad={handleImageLoad}
          />
        </div>
      </div>
    </div>
  );
}
