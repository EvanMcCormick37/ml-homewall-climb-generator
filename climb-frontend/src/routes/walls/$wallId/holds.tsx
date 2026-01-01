import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl, setHolds } from "@/api/walls";
import { useHolds } from "@/hooks/useHolds";
import { HoldFeaturesSidebar, EnabledFeaturesMenu } from "@/components";
import { Eraser, PlusCircle, Hand, Settings, Plus, Edit } from "lucide-react";
import type { HoldDetail, WallDetail, HoldMode } from "@/types";

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
  const wallDimensions = {
    width: wall.metadata.dimensions[0],
    height: wall.metadata.dimensions[1],
  };

  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [mode, setHoldMode] = useState<HoldMode>("add");
  const [selectedHold, setSelectedHold] = useState<HoldDetail | null>(null);
  const [isAddFoot, setIsAddFoot] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewTransform, setViewTransform] = useState({ zoom: 1, x: 0, y: 0 });
  const [showFeatureMenu, setShowFeatureMenu] = useState(false);
  const [enabledFeatures, setEnabledFeatures] = useState({
    direction: true,
    useability: true,
    footholds: true,
  });
  const [useabilityLocked, setUseabilityLocked] = useState(false);
  const [lockedUseability, setLockedUseability] = useState(0.5);

  const {
    holds,
    addHold,
    updateHold,
    removeHold,
    removeHoldByIndex,
    removeLastHold,
    findHoldAt,
    clearHolds,
    loadHolds,
    toPixelCoords,
    toFeetCoords,
  } = useHolds(imageDimensions, wallDimensions);

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

  const [editHoldState, setEditHoldState] = useState<{
    isDragging: boolean;
    dragX: number;
    dragY: number;
    originalHold?: HoldDetail;
    useability?: number;
    is_foot?: number;
  }>({
    isDragging: false,
    dragX: 0,
    dragY: 0,
  });

  // Toggle feature in menu
  const handleFeatureToggle = (feature: keyof typeof enabledFeatures) => {
    setEnabledFeatures((prev) => ({
      ...prev,
      [feature]: !prev[feature],
    }));
  };

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
    if (wall.holds && imageDimensions.width > 0) {
      loadHolds(wall.holds);
    }
  }, [wall.holds, loadHolds, imageDimensions.width]);

  // Canvas Utility - get image coordinates from mouse event
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

  // Calculate hold params from drag gesture (in pixels, then convert to feet for display)
  const calculateHoldParams = useCallback(
    (holdX: number, holdY: number, dragX: number, dragY: number) => {
      const dx = holdX - dragX;
      const dy = holdY - dragY;
      const magnitude = Math.sqrt(dx * dx + dy * dy);
      const feetCoords = toFeetCoords(holdX, holdY);

      return {
        pull_x: magnitude === 0 ? 0 : dx / magnitude,
        pull_y: magnitude === 0 ? -1 : dy / magnitude,
        useability: enabledFeatures.useability
          ? useabilityLocked
            ? lockedUseability
            : Math.min(1, magnitude / 250)
          : 0.5,
        x: feetCoords.x,
        y: feetCoords.y,
      };
    },
    [toFeetCoords, enabledFeatures, lockedUseability, useabilityLocked]
  );

  // Get color based on hold type
  const getHoldColor = useCallback((u: number, isFoot: boolean) => {
    if (isFoot) {
      // Dark Violet -> Light Turquoise
      const r = Math.round(60 - 60 * u);
      const g = Math.round(0 + 200 * u);
      const b = Math.round(40 + 140 * u);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Original color scheme
      const r = u < 0.5 ? 255 : Math.round(60 + 195 * (1 - u) * 2);
      const g = u < 0.5 ? Math.round(60 + 160 * u * 2) : 220;
      return `rgb(${r}, ${g}, 60)`;
    }
  }, []);

  // Legacy function for backward compatibility
  const getUseabilityColor = useCallback(
    (u: number) => getHoldColor(u, false),
    [getHoldColor]
  );

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
    if (mode === "add") {
      // If neither direction nor useability is enabled, add hold immediately
      if (!enabledFeatures.direction && !enabledFeatures.useability) {
        addHold(
          x,
          y,
          undefined, // default hold index
          undefined, // no pull_x
          undefined, // no pull_y
          undefined, // no useability
          enabledFeatures.footholds && isAddFoot ? 1 : 0
        );
      } else {
        // Start drag for direction/useability
        setAddHoldState({
          isDragging: true,
          holdX: x,
          holdY: y,
          dragX: x,
          dragY: y,
        });
      }
    } else if (mode === "edit") {
      // In edit mode, only respond if clicking on an existing hold
      const hold = findHoldAt(x, y);
      if (hold) {
        const pixelCoords = toPixelCoords(hold);
        setEditHoldState({
          isDragging: true,
          dragX: pixelCoords.x,
          dragY: pixelCoords.y,
          originalHold: { ...hold },
        });
      }
      // If clicking on empty space, do nothing
    } else if (mode === "remove") {
      removeHold(x, y);
    } else if (mode === "select") {
      const hold = findHoldAt(x, y);
      setSelectedHold(hold);
    }
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
    } else if (editHoldState.isDragging) {
      const { x, y } = getImageCoords(e);
      setEditHoldState((prev) => ({ ...prev, dragX: x, dragY: y }));
    }
  };

  const handleMouseUp = () => {
    if (panDragRef.current.isDragging) panDragRef.current.isDragging = false;
    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const params = calculateHoldParams(holdX, holdY, dragX, dragY);

      addHold(
        holdX,
        holdY,
        undefined,
        enabledFeatures.direction ? params.pull_x : undefined,
        enabledFeatures.direction ? params.pull_y : undefined,
        enabledFeatures.useability ? params.useability : undefined,
        enabledFeatures.footholds && isAddFoot ? 1 : 0
      );
      setAddHoldState({
        isDragging: false,
        holdX: 0,
        holdY: 0,
        dragX: 0,
        dragY: 0,
      });
    } else if (editHoldState.isDragging && editHoldState.originalHold) {
      // In edit mode, update the hold's properties but keep its original position
      const originalHold = editHoldState.originalHold;
      const pixelCoords = toPixelCoords(originalHold);
      const { dragX, dragY } = editHoldState;

      // Calculate new properties based on drag from hold position
      const updatedParams = {
        ...calculateHoldParams(pixelCoords.x, pixelCoords.y, dragX, dragY),
        is_foot: Number(isAddFoot),
      };

      // update the hold
      updateHold(originalHold.hold_index, updatedParams);

      setEditHoldState({
        isDragging: false,
        dragX: 0,
        dragY: 0,
      });
    }
  };

  // Scroll wheel
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

  // Hotkeys
  useEffect(() => {
    const handleKeydown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        removeLastHold();
      }
      const key = e.key;
      switch (key) {
        case "1":
          e.preventDefault();
          setHoldMode("add");
          break;
        case "2":
          e.preventDefault();
          setHoldMode("remove");
          break;
        case "3":
          e.preventDefault();
          setHoldMode("select");
          break;
        case "x":
        case "X":
          setIsAddFoot((prev) => !prev);
          break;
      }
    };
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [removeLastHold]);

  // Canvas Rendering
  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas?.getContext("2d");
    if (!image || !ctx) return;

    const { width, height } = imageDimensions;
    canvas.width = width || 800;
    canvas.height = height || 600;

    ctx.fillStyle = "#18181b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    // Draw holds
    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const isFoot = !!hold.is_foot;
      const sizeMultiplier = isFoot ? 0.6 : 1;
      const useability = hold.useability ?? 0.5;
      const color = getHoldColor(useability, isFoot);

      // Hold Display Constants
      const circleSize = 4 * sizeMultiplier;
      const arrowSize = 3 * sizeMultiplier;

      // Selection highlight
      if (mode === "select" && selectedHold?.hold_index === hold.hold_index) {
        ctx.beginPath();
        ctx.arc(x, y, 20 * sizeMultiplier, 0, Math.PI * 2);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 4 * sizeMultiplier;
        ctx.stroke();
      }

      // Hold circle
      ctx.beginPath();
      ctx.arc(x, y, circleSize * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = circleSize;
      ctx.stroke();

      // Pull direction arrow (only if direction data exists)
      if (
        hold.pull_x !== null &&
        hold.pull_x !== undefined &&
        hold.pull_y !== null &&
        hold.pull_y !== undefined
      ) {
        const arrowLength = (10 + 30 * arrowSize * useability) * sizeMultiplier;
        const endX = x + hold.pull_x * arrowLength;
        const endY = y + hold.pull_y * arrowLength;

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = color;
        ctx.lineWidth = arrowSize;
        ctx.stroke();

        const headLength = arrowLength / 5.0;
        const angle = Math.atan2(hold.pull_y, hold.pull_x);
        ctx.beginPath();
        ctx.moveTo(
          endX + (arrowSize / 2.0) * Math.cos(angle - Math.PI / 4),
          endY + (arrowSize / 2.0) * Math.sin(angle - Math.PI / 4)
        );
        ctx.lineTo(
          endX - headLength * Math.cos(angle - Math.PI / 4),
          endY - headLength * Math.sin(angle - Math.PI / 4)
        );
        ctx.moveTo(
          endX + (arrowSize / 2.0) * Math.cos(angle + Math.PI / 4),
          endY + (arrowSize / 2.0) * Math.sin(angle + Math.PI / 4)
        );
        ctx.lineTo(
          endX - headLength * Math.cos(angle + Math.PI / 4),
          endY - headLength * Math.sin(angle + Math.PI / 4)
        );
        ctx.stroke();
      }

      // Hold index label
      ctx.fillStyle = "white";
      ctx.font = `bold ${10 * sizeMultiplier}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(hold.hold_index.toString(), x, y);
    });

    // Draw preview while dragging
    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const params = calculateHoldParams(holdX, holdY, dragX, dragY);
      const sizeMultiplier = isAddFoot ? 0.5 : 1;
      // Use locked useability if enabled, otherwise use calculated value
      const color = getHoldColor(params.useability, isAddFoot);

      const circleSize = 6 * sizeMultiplier;
      const arrowSize = 4 * sizeMultiplier;

      ctx.beginPath();
      ctx.arc(holdX, holdY, circleSize * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = circleSize;
      ctx.stroke();
      ctx.setLineDash([]);

      // Only draw arrow if direction is enabled
      if (enabledFeatures.direction) {
        ctx.beginPath();
        ctx.moveTo(dragX, dragY);
        ctx.lineTo(holdX, holdY);
        ctx.strokeStyle = color;
        ctx.lineWidth = arrowSize;
        ctx.stroke();
      }
    }

    // Draw preview while editing
    if (editHoldState.isDragging && editHoldState.originalHold) {
      const originalHold = editHoldState.originalHold;
      const pixelCoords = toPixelCoords(originalHold);
      const { dragX, dragY } = editHoldState;
      const params = calculateHoldParams(
        pixelCoords.x,
        pixelCoords.y,
        dragX,
        dragY
      );
      const sizeMultiplier = originalHold.is_foot ? 0.5 : 1;
      const color = getHoldColor(params.useability, originalHold.is_foot === 1);

      const circleSize = 6 * sizeMultiplier;
      const arrowSize = 4 * sizeMultiplier;

      ctx.beginPath();
      ctx.arc(pixelCoords.x, pixelCoords.y, circleSize * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = circleSize;
      ctx.stroke();
      ctx.setLineDash([]);

      // Only draw arrow if direction is enabled
      if (enabledFeatures.direction) {
        ctx.beginPath();
        ctx.moveTo(dragX, dragY);
        ctx.lineTo(pixelCoords.x, pixelCoords.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = arrowSize;
        ctx.stroke();
      }
    }
  }, [
    image,
    imageDimensions,
    holds,
    addHoldState,
    editHoldState,
    calculateHoldParams,
    getHoldColor,
    mode,
    selectedHold,
    toPixelCoords,
    enabledFeatures,
    isAddFoot,
  ]);

  // Drag params for sidebar
  const dragParams = addHoldState.isDragging
    ? {
        ...calculateHoldParams(
          addHoldState.holdX,
          addHoldState.holdY,
          addHoldState.dragX,
          addHoldState.dragY
        ),
      }
    : editHoldState.isDragging && editHoldState.originalHold
      ? {
          ...(() => {
            const pixelCoords = toPixelCoords(editHoldState.originalHold);
            return calculateHoldParams(
              pixelCoords.x,
              pixelCoords.y,
              editHoldState.dragX,
              editHoldState.dragY
            );
          })(),
        }
      : { pull_x: 0, pull_y: -1, useability: 0, x: 0, y: 0 };

  return (
    <div className="relative h-[calc(100vh-4rem)] flex flex-col bg-zinc-950 overflow-hidden">
      {/* Feature Selection Menu */}
      {showFeatureMenu && (
        <EnabledFeaturesMenu
          enabledFeatures={enabledFeatures}
          onToggle={handleFeatureToggle}
          onClose={() => setShowFeatureMenu(false)}
        />
      )}

      {/* Header Toolbar */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900 z-10">
        <button
          onClick={() => setShowFeatureMenu(!showFeatureMenu)}
          className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${showFeatureMenu ? "bg-purple-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
        >
          <Settings size={14} />
        </button>
        <div className="flex items-center gap-6">
          <h1 className="text-sm font-bold text-zinc-400 uppercase tracking-widest">
            {wall.metadata.name} <span className="text-zinc-600">/ Editor</span>
          </h1>
          <div className="flex bg-zinc-950 rounded-lg p-1 border border-zinc-800">
            <button
              onClick={() => setHoldMode("add")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${mode === "add" ? (enabledFeatures.footholds && isAddFoot ? "bg-purple-600 text-white" : "bg-emerald-600 text-white") : "text-zinc-500 hover:text-zinc-300"}`}
            >
              {mode === "add" ? (
                <span className="flex">
                  <Plus size={14} />
                  {`${
                    enabledFeatures.footholds && isAddFoot ? "FOOT" : "HAND"
                  }`}
                </span>
              ) : (
                <PlusCircle size={14} />
              )}
            </button>
            <button
              onClick={() => setHoldMode("edit")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all flex items-center gap-2 ${mode === "edit" ? "bg-amber-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
            >
              <Edit size={14} />
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
                await setHolds(wallId, holds);
                navigate({ to: `/walls/${wallId}` });
              } catch (err) {
                setError(
                  err instanceof Error ? err.message : "Failed to save holds"
                );
              } finally {
                setIsSubmitting(false);
              }
            }}
            disabled={isSubmitting}
            className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded-lg transition-all disabled:opacity-50"
          >
            {isSubmitting ? "SAVING..." : "SAVE HOLDS"}
          </button>
        </div>
      </header>

      {error && (
        <div className="px-6 py-2 bg-red-900/50 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Canvas area */}
        <div
          ref={wrapperRef}
          className="flex-1 overflow-hidden bg-zinc-950 cursor-crosshair"
        >
          <div
            style={{
              transform: `translate(${viewTransform.x}px, ${viewTransform.y}px)`,
            }}
          >
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{
                width: (imageDimensions.width || 800) * viewTransform.zoom,
                height: (imageDimensions.height || 600) * viewTransform.zoom,
              }}
            />
          </div>
        </div>

        {/* Sidebar */}
        <HoldFeaturesSidebar
          mode={mode}
          enabledFeatures={enabledFeatures}
          selectedHold={selectedHold}
          isDragging={addHoldState.isDragging || editHoldState.isDragging}
          dragParams={dragParams}
          getColor={getUseabilityColor}
          onDeleteHold={() => {
            if (selectedHold) {
              removeHoldByIndex(selectedHold.hold_index);
              setSelectedHold(null);
            }
          }}
          useabilityLocked={useabilityLocked}
          lockedUseability={lockedUseability}
          onUseabilityLockChange={setUseabilityLocked}
          onLockedUseabilityChange={setLockedUseability}
        />
      </div>
    </div>
  );
}
