import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl, setHolds } from "@/api/walls";
import { useHolds } from "@/hooks/useHolds";
import { HelpOverlay, HoldFeaturesSidebar } from "@/components";
import { Eraser, PlusCircle, Hand } from "lucide-react";
import type { HoldWithPixels, WallDetail, HoldMode } from "@/types";

// --- Main Page ---

export const Route = createFileRoute("/walls/$wallId/holds")({
  component: HoldsEditorPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

function HoldsEditorPage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const wallId = wall.metadata.id;

  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [mode, setHoldMode] = useState<HoldMode>("add");
  const [selectedHold, setSelectedHold] = useState<HoldWithPixels | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewTransform, setViewTransform] = useState({ zoom: 1, x: 0, y: 0 });

  const {
    holds,
    addHold,
    removeHold,
    removeHoldById,
    removeLastHold,
    findHoldAt,
    clearHolds,
    loadHolds,
  } = useHolds(imageDimensions);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
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

  // Load image and setup camera
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImage(img);
      setImageDimensions({ width: img.width, height: img.height });
      if (wrapperRef.current) {
        const rect = wrapperRef.current.getBoundingClientRect();
        const scale =
          Math.min(rect.width / img.width, rect.height / img.height) * 0.9;
        setViewTransform({
          zoom: scale,
          x: (rect.width - img.width * scale) / 2,
          y: (rect.height - img.height * scale) / 2,
        });
      }
    };
    img.src = getWallPhotoUrl(wallId);
  }, [wallId]);

  useEffect(() => {
    if (wall.holds) loadHolds({ holds: wall.holds });
  }, [wall.holds, loadHolds]);

  // Canvas Utility
  const getImageCoords = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.round(
          (e.clientX - rect.left) * (imageDimensions.width / rect.width)
        ),
        y: Math.round(
          (e.clientY - rect.top) * (imageDimensions.height / rect.height)
        ),
      };
    },
    [imageDimensions]
  );

  const calculateHoldParams = useCallback(
    (holdX: number, holdY: number, dragX: number, dragY: number) => {
      const dx = holdX - dragX;
      const dy = holdY - dragY;
      const magnitude = Math.sqrt(dx * dx + dy * dy);
      return {
        pull_x: magnitude === 0 ? 0 : dx / magnitude,
        pull_y: magnitude === 0 ? -1 : dy / magnitude,
        useability: Math.min(1, magnitude / 250),
        norm_x: holdX / imageDimensions.width,
        norm_y: 1 - holdY / imageDimensions.height,
      };
    },
    [imageDimensions]
  );

  const getUseabilityColor = useCallback((u: number) => {
    const r = u < 0.5 ? 255 : Math.round(60 + 195 * (1 - u) * 2);
    const g = u < 0.5 ? Math.round(60 + 160 * u * 2) : 220;
    return `rgb(${r}, ${g}, 60)`;
  }, []);

  // Handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getImageCoords(e);
    if (e.button === 1 || e.shiftKey) {
      panDragRef.current = {
        isDragging: true,
        startX: e.clientX,
        startY: e.clientY,
        startViewX: viewTransform.x,
        startViewY: viewTransform.y,
      };
      return;
    }
    if (mode === "add")
      setAddHoldState({
        isDragging: true,
        holdX: x,
        holdY: y,
        dragX: x,
        dragY: y,
      });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
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
    } else if (addHoldState.isDragging) {
      const { x, y } = getImageCoords(e);
      setAddHoldState((prev) => ({ ...prev, dragX: x, dragY: y }));
    }
  };

  const handleMouseUp = () => {
    if (panDragRef.current.isDragging) panDragRef.current.isDragging = false;
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
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const rect = wrapperRef.current?.getBoundingClientRect();
    if (!rect) return;

    // Zoom toward cursor position
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    setViewTransform((prev) => {
      const newZoom = Math.min(Math.max(prev.zoom * zoomFactor, 0.1), 5);
      const scale = newZoom / prev.zoom;
      return {
        zoom: newZoom,
        x: mouseX - (mouseX - prev.x) * scale,
        y: mouseY - (mouseY - prev.y) * scale,
      };
    });
  };

  const handleKeydown = useCallback((e: KeyboardEvent) => {
    const key = e.key;
    switch (key) {
      case "1":
        setHoldMode("add");
        break;
      case "2":
        setHoldMode("remove");
        break;
      case "3":
        setHoldMode("select");
        break;
      case "Backspace":
        removeLastHold();
        break;
    }
  }, []);

  useEffect(() => {
    const globalHandleKeydown = (e: KeyboardEvent) => handleKeydown(e);
    window.addEventListener("keydown", globalHandleKeydown);
    return () => {
      window.removeEventListener("keydown", globalHandleKeydown);
    };
  }, []);

  // Rendering logic (Canvas effect omitted for brevity, same as previous but using calculateHoldParams)
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !image) return;
    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;
    ctx.drawImage(image, 0, 0);

    holds.forEach((hold) => {
      const x = hold.norm_x * imageDimensions.width;
      const y = (1 - hold.norm_y) * imageDimensions.height;
      const color = getUseabilityColor(hold.useability ?? 0.5);
      if (mode === "select" && selectedHold?.hold_id === hold.hold_id) {
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, Math.PI * 2);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 4;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();
      // Draw arrow... (logic from original code)
    });

    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const { pull_x, pull_y, useability } = calculateHoldParams(
        holdX,
        holdY,
        dragX,
        dragY
      );
      const color = getUseabilityColor(useability);
      ctx.beginPath();
      ctx.arc(holdX, holdY, 15, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(dragX, dragY);
      ctx.lineTo(holdX, holdY);
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }, [
    image,
    imageDimensions,
    holds,
    addHoldState,
    calculateHoldParams,
    getUseabilityColor,
    mode,
    selectedHold,
  ]);

  return (
    <div className="relative h-[calc(100vh-4rem)] flex flex-col bg-zinc-950 overflow-hidden">
      <HelpOverlay />

      {/* Header Toolbar */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900 z-10">
        <div className="flex items-center gap-6">
          <h1 className="text-sm font-bold text-zinc-400 uppercase tracking-widest">
            {wall.metadata.name} <span className="text-zinc-600">/ Editor</span>
          </h1>
          <div className="flex bg-zinc-950 rounded-lg p-1 border border-zinc-800">
            <button
              onClick={() => setHoldMode("add")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${mode === "add" ? "bg-emerald-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
            >
              <PlusCircle size={14} />
            </button>
            <button
              onClick={() => setHoldMode("remove")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${mode === "remove" ? "bg-red-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
            >
              <Eraser size={14} />
            </button>
            <button
              onClick={() => setHoldMode("select")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${mode === "select" ? "bg-blue-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
            >
              <Hand size={14} />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              clearHolds();
              setSelectedHold(null);
            }}
            className="px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 text-zinc-400 hover:bg-red-600 hover:text-white"
          >
            CLEAR HOLDS
          </button>
          <button
            onClick={() => navigate({ to: `/walls/${wallId}` })}
            className="px-4 py-2 text-xs font-bold text-zinc-500 hover:text-zinc-300"
          >
            CANCEL
          </button>
          <button
            onClick={async () => {
              setIsSubmitting(true);
              setError(null);
              try {
                // Strip pixel coordinates, only send normalized data
                const holdsToSubmit = holds.map(
                  ({
                    hold_id,
                    norm_x,
                    norm_y,
                    pull_x,
                    pull_y,
                    useability,
                  }) => ({
                    hold_id,
                    norm_x,
                    norm_y,
                    pull_x: pull_x ?? 0,
                    pull_y: pull_y ?? -1,
                    useability: useability ?? 0.5,
                  })
                );
                await setHolds(wallId, holdsToSubmit);
                navigate({ to: `/walls/${wallId}` });
              } catch (err) {
                setError(
                  err instanceof Error ? err.message : "Failed to save holds"
                );
              } finally {
                setIsSubmitting(false);
              }
            }}
            className="px-6 py-2 hover:bg-indigo-500 text-white text-xs font-black rounded-lg transition-all"
          >
            SUBMIT HOLDS
          </button>
        </div>
      </header>

      {/* Editor Body */}
      <div className="flex-1 flex overflow-hidden">
        <main
          className="flex-1 bg-zinc-950 relative overflow-hidden"
          ref={wrapperRef}
          onWheel={handleWheel}
        >
          <div
            style={{
              transform: `translate(${viewTransform.x}px, ${viewTransform.y}px)`,
            }}
          >
            <canvas
              ref={canvasRef}
              style={{
                width: imageDimensions.width * viewTransform.zoom,
                height: imageDimensions.height * viewTransform.zoom,
                cursor: mode === "add" ? "crosshair" : "pointer",
              }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onClick={(e) => {
                const { x, y } = getImageCoords(e);
                if (mode === "remove") removeHold(x, y);
                if (mode === "select") setSelectedHold(findHoldAt(x, y));
              }}
            />
          </div>
        </main>

        <HoldFeaturesSidebar
          mode={mode}
          selectedHold={selectedHold}
          isDragging={addHoldState.isDragging}
          dragParams={calculateHoldParams(
            addHoldState.holdX,
            addHoldState.holdY,
            addHoldState.dragX,
            addHoldState.dragY
          )}
          getColor={getUseabilityColor}
          onDeleteHold={() => {
            if (selectedHold) {
              removeHoldById(selectedHold.hold_id);
              setSelectedHold(null);
            }
          }}
        />
      </div>
    </div>
  );
}
