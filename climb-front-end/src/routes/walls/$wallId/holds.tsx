import { createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  useState,
  useCallback,
  useRef,
  useEffect,
  useMemo,
} from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { useHolds } from "@/hooks/useHolds";
import { apiClient } from "@/api/client";
import type { HoldDetail, WallDetail } from "@/types";

export const Route = createFileRoute("/walls/$wallId/holds")({
  component: HoldsEditorPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

type Mode = "add" | "remove" | "pan";

// Icon components for toolbar
function PlusIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="16" />
      <line x1="8" y1="12" x2="16" y2="12" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function MoveIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="5 9 2 12 5 15" />
      <polyline points="9 5 12 2 15 5" />
      <polyline points="15 19 12 22 9 19" />
      <polyline points="19 9 22 12 19 15" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <line x1="12" y1="2" x2="12" y2="22" />
    </svg>
  );
}

function HoldsEditorPage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const wallId = wall.metadata.id;

  // Image state
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Mode state
  const [mode, setMode] = useState<Mode>("add");

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // View transform state (pan/zoom)
  const [viewTransform, setViewTransform] = useState({
    zoom: 1,
    x: 0,
    y: 0,
  });

  // Holds state from hook
  const {
    holds,
    addHold,
    removeHold,
    findHoldAt,
    clearHolds,
    loadHolds,
  } = useHolds(imageDimensions);

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Drag states
  const panDragRef = useRef({
    isDragging: false,
    startX: 0,
    startY: 0,
    startViewX: 0,
    startViewY: 0,
  });

  const [addHoldState, setAddHoldState] = useState({
    isDragging: false,
    holdX: 0,
    holdY: 0,
    dragX: 0,
    dragY: 0,
  });

  // Load wall image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImage(img);
      setImageDimensions({ width: img.width, height: img.height });

      // Auto-fit to container
      if (wrapperRef.current) {
        const rect = wrapperRef.current.getBoundingClientRect();
        const scaleX = rect.width / img.width;
        const scaleY = rect.height / img.height;
        const newZoom = Math.min(scaleX, scaleY) * 0.95;
        setViewTransform({
          zoom: newZoom,
          x: (rect.width - img.width * newZoom) / 2,
          y: (rect.height - img.height * newZoom) / 2,
        });
      }
    };
    img.src = getWallPhotoUrl(wallId);
  }, [wallId]);

  // Load existing holds
  useEffect(() => {
    if (wall.holds && wall.holds.length > 0) {
      loadHolds({ holds: wall.holds });
    }
  }, [wall.holds, loadHolds]);

  // Get image coordinates from mouse event
  const getImageCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };

      const rect = canvas.getBoundingClientRect();
      const scaleX = imageDimensions.width / rect.width;
      const scaleY = imageDimensions.height / rect.height;

      return {
        x: Math.round((e.clientX - rect.left) * scaleX),
        y: Math.round((e.clientY - rect.top) * scaleY),
      };
    },
    [imageDimensions]
  );

  // Calculate pull direction and useability from drag vector
  // Useability is now [0, 1] instead of [1, 10]
  const calculateHoldParams = useCallback(
    (holdX: number, holdY: number, dragX: number, dragY: number) => {
      const dx = holdX - dragX;
      const dy = holdY - dragY;
      const magnitude = Math.sqrt(dx * dx + dy * dy);

      if (magnitude === 0) {
        return { pull_x: 0, pull_y: -1, useability: 0.5 };
      }

      const pull_x = dx / magnitude;
      const pull_y = dy / magnitude;
      // Map magnitude to [0, 1]: 0px = 0, 250px+ = 1
      const useability = Math.min(1, magnitude / 250);

      return { pull_x, pull_y, useability };
    },
    []
  );

  // Get color based on useability (0-1 scale)
  const getUseabilityColor = useCallback((useability: number) => {
    // Red (hard) -> Yellow -> Green (easy)
    const t = useability;
    let r: number, g: number, b: number;
    if (t < 0.5) {
      const t2 = t * 2;
      r = 255;
      g = Math.round(60 + 160 * t2);
      b = 60;
    } else {
      const t2 = (1 - t) * 2;
      r = Math.round(60 + 195 * t2);
      g = 220;
      b = 60;
    }
    return `rgb(${r}, ${g}, ${b})`;
  }, []);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !image) return;

    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;

    // Clear and draw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    // Draw holds
    holds.forEach((hold) => {
      const x = hold.norm_x * imageDimensions.width;
      const y = (1 - hold.norm_y) * imageDimensions.height;
      const color = getUseabilityColor(hold.useability ?? 0.5);

      // Draw hold circle
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Draw pull direction arrow
      if (hold.pull_x !== undefined && hold.pull_y !== undefined) {
        const arrowLength = 25 + (hold.useability ?? 0.5) * 25;
        const endX = x + hold.pull_x * arrowLength;
        const endY = y + hold.pull_y * arrowLength;

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(hold.pull_y, hold.pull_x);
        const headLength = 8;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - headLength * Math.cos(angle - Math.PI / 6),
          endY - headLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - headLength * Math.cos(angle + Math.PI / 6),
          endY - headLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
      }

      // Draw hold ID
      ctx.font = "bold 10px sans-serif";
      ctx.fillStyle = "white";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(hold.hold_id), x, y);
    });

    // Draw add-hold preview if dragging
    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const { pull_x, pull_y, useability } = calculateHoldParams(
        holdX,
        holdY,
        dragX,
        dragY
      );
      const dragColor = getUseabilityColor(useability);

      // Preview circle
      ctx.beginPath();
      ctx.arc(holdX, holdY, 15, 0, 2 * Math.PI);
      ctx.strokeStyle = dragColor;
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Preview line
      ctx.beginPath();
      ctx.moveTo(dragX, dragY);
      ctx.lineTo(holdX, holdY);
      ctx.strokeStyle = dragColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Preview arrow head
      const angle = Math.atan2(holdY - dragY, holdX - dragX);
      const headLength = 12;
      ctx.beginPath();
      ctx.moveTo(holdX, holdY);
      ctx.lineTo(
        holdX - headLength * Math.cos(angle - Math.PI / 6),
        holdY - headLength * Math.sin(angle - Math.PI / 6)
      );
      ctx.moveTo(holdX, holdY);
      ctx.lineTo(
        holdX - headLength * Math.cos(angle + Math.PI / 6),
        holdY - headLength * Math.sin(angle + Math.PI / 6)
      );
      ctx.stroke();

      // Useability indicator
      ctx.font = "bold 14px sans-serif";
      ctx.fillStyle = dragColor;
      ctx.textAlign = "left";
      ctx.fillText(`Useability: ${(useability * 100).toFixed(0)}%`, dragX + 10, dragY - 10);
    }
  }, [
    image,
    imageDimensions,
    holds,
    addHoldState,
    calculateHoldParams,
    getUseabilityColor,
  ]);

  // Mouse handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!image) return;

      const { x, y } = getImageCoords(e);

      if (mode === "pan") {
        panDragRef.current = {
          isDragging: true,
          startX: e.clientX,
          startY: e.clientY,
          startViewX: viewTransform.x,
          startViewY: viewTransform.y,
        };
      } else if (mode === "add") {
        setAddHoldState({
          isDragging: true,
          holdX: x,
          holdY: y,
          dragX: x,
          dragY: y,
        });
      }
    },
    [image, mode, getImageCoords, viewTransform]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      // Pan mode dragging
      if (panDragRef.current.isDragging) {
        const drag = panDragRef.current;
        setViewTransform((prev) => ({
          ...prev,
          x: drag.startViewX + (e.clientX - drag.startX),
          y: drag.startViewY + (e.clientY - drag.startY),
        }));
        return;
      }

      // Add mode dragging
      if (addHoldState.isDragging) {
        const { x, y } = getImageCoords(e);
        setAddHoldState((prev) => ({
          ...prev,
          dragX: x,
          dragY: y,
        }));
      }
    },
    [addHoldState.isDragging, getImageCoords]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (panDragRef.current.isDragging) {
        panDragRef.current.isDragging = false;
        return;
      }

      if (addHoldState.isDragging) {
        const { holdX, holdY, dragX, dragY } = addHoldState;
        const { pull_x, pull_y, useability } = calculateHoldParams(
          holdX,
          holdY,
          dragX,
          dragY
        );

        addHold(holdX, holdY, pull_x, pull_y, useability);

        setAddHoldState({
          isDragging: false,
          holdX: 0,
          holdY: 0,
          dragX: 0,
          dragY: 0,
        });
      }
    },
    [addHoldState, calculateHoldParams, addHold]
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!image || mode !== "remove") return;

      const { x, y } = getImageCoords(e);
      removeHold(x, y);
    },
    [image, mode, getImageCoords, removeHold]
  );

  // Wheel zoom handler
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const wrapper = wrapperRef.current;
      if (!wrapper) return;

      const rect = wrapper.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      setViewTransform((prev) => {
        const newZoom = Math.max(0.1, Math.min(5, prev.zoom * delta));
        const beforeX = (mouseX - prev.x) / prev.zoom;
        const beforeY = (mouseY - prev.y) / prev.zoom;
        return {
          zoom: newZoom,
          x: mouseX - beforeX * newZoom,
          y: mouseY - beforeY * newZoom,
        };
      });
    },
    []
  );

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    wrapper.addEventListener("wheel", handleWheel, { passive: false });
    return () => wrapper.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // Cursor based on mode
  const getCursor = () => {
    if (addHoldState.isDragging) return "crosshair";
    if (panDragRef.current.isDragging) return "grabbing";

    switch (mode) {
      case "pan":
        return "grab";
      case "add":
        return "crosshair";
      case "remove":
        return "pointer";
      default:
        return "default";
    }
  };

  // Submit holds to API
  const handleSubmit = useCallback(async () => {
    setIsSubmitting(true);
    setError(null);

    try {
      // Prepare holds for API - renumber IDs sequentially
      const apiHolds: HoldDetail[] = holds.map((hold, idx) => ({
        hold_id: idx,
        norm_x: hold.norm_x,
        norm_y: hold.norm_y,
        pull_x: hold.pull_x ?? 0,
        pull_y: hold.pull_y ?? -1,
        useability: hold.useability ?? 0.5,
      }));

      const formData = new FormData();
      formData.append("holds", JSON.stringify(apiHolds));

      await apiClient.put(`/walls/${wallId}/holds`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      // Navigate to wall overview on success
      navigate({ to: `/walls/${wallId}` });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save holds";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [holds, wallId, navigate]);

  const { zoom, x, y } = viewTransform;
  const { width, height } = imageDimensions;

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col bg-zinc-950">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900">
        <div className="flex items-center gap-6">
          <h1 className="text-lg font-medium text-zinc-100">
            Edit Holds: {wall.metadata.name}
          </h1>

          {/* Mode selector */}
          <div className="flex items-center gap-1 bg-zinc-800 rounded-lg p-1">
            <button
              onClick={() => setMode("add")}
              className={`p-2 rounded-md transition-colors ${
                mode === "add"
                  ? "bg-emerald-600 text-white"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
              }`}
              title="Add Hold (click and drag to set position and pull direction)"
            >
              <PlusIcon className="w-5 h-5" />
            </button>
            <button
              onClick={() => setMode("remove")}
              className={`p-2 rounded-md transition-colors ${
                mode === "remove"
                  ? "bg-red-600 text-white"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
              }`}
              title="Remove Hold (click on a hold to delete)"
            >
              <TrashIcon className="w-5 h-5" />
            </button>
            <button
              onClick={() => setMode("pan")}
              className={`p-2 rounded-md transition-colors ${
                mode === "pan"
                  ? "bg-blue-600 text-white"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
              }`}
              title="Pan (drag to move view)"
            >
              <MoveIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Hold count */}
          <span className="text-sm text-zinc-500">
            {holds.length} hold{holds.length !== 1 ? "s" : ""}
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Clear button */}
          <button
            onClick={() => {
              if (window.confirm("Remove all holds?")) {
                clearHolds();
              }
            }}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            Clear All
          </button>

          {/* Cancel button */}
          <button
            onClick={() => navigate({ to: `/walls/${wallId}` })}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            Cancel
          </button>

          {/* Submit button */}
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || holds.length === 0}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg font-medium transition-colors"
          >
            {isSubmitting ? "Saving..." : "Submit Holds"}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="px-6 py-3 bg-red-900/50 border-b border-red-800 text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Canvas area */}
      <div
        className="flex-1 overflow-hidden bg-zinc-900"
        ref={wrapperRef}
      >
        <div
          className="relative"
          style={{
            transform: `translate(${x}px, ${y}px)`,
          }}
        >
          <canvas
            ref={canvasRef}
            style={{
              width: (width || 800) * zoom,
              height: (height || 600) * zoom,
              cursor: getCursor(),
            }}
            onClick={handleClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => {
              if (panDragRef.current.isDragging) {
                panDragRef.current.isDragging = false;
              }
              if (addHoldState.isDragging) {
                setAddHoldState((prev) => ({ ...prev, isDragging: false }));
              }
            }}
          />
        </div>
      </div>

      {/* Help text */}
      <div className="px-6 py-3 border-t border-zinc-800 bg-zinc-900 text-zinc-500 text-xs">
        <span className="font-medium text-zinc-400">Add:</span> Click and drag to place hold with pull direction (drag length = useability) •{" "}
        <span className="font-medium text-zinc-400">Remove:</span> Click on hold to delete •{" "}
        <span className="font-medium text-zinc-400">Pan:</span> Drag to move view •{" "}
        <span className="font-medium text-zinc-400">Zoom:</span> Scroll wheel
      </div>
    </div>
  );
}
