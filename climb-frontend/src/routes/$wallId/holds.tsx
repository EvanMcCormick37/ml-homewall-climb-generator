import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getWall, getWallPhotoUrl, setHolds } from "@/api/walls";
import { useHolds } from "@/hooks/useHolds";
import { HoldFeaturesSidebar, EnabledFeaturesMenu } from "@/components";
import { Eraser, Hand, Settings, Plus, Edit } from "lucide-react";
import { GLOBAL_STYLES } from "@/styles";
import type { HoldDetail, WallDetail, HoldMode, Tag } from "@/types";

export const Route = createFileRoute("/$wallId/holds")({
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
    tags: true,
  });
  const [useabilityLocked, setUseabilityLocked] = useState(false);
  const [lockedUseability, setLockedUseability] = useState(0.5);
  const [activeHoldIndex, setActiveHoldIndex] = useState<number | null>(null);
  const [stickyTags, setStickyTags] = useState<Tag[]>([]);

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

  const activeHold =
    activeHoldIndex !== null
      ? (holds.find((h) => h.hold_index === activeHoldIndex) ?? null)
      : null;

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

  const handleFeatureToggle = (feature: keyof typeof enabledFeatures) => {
    setEnabledFeatures((prev) => ({
      ...prev,
      [feature]: !prev[feature],
    }));
  };

  function handleTagToggle(tag: Tag) {
    if (activeHoldIndex === null) return;
    const hold = holds.find((h) => h.hold_index === activeHoldIndex);
    if (!hold) return;
    const newTags = hold.tags.includes(tag)
      ? hold.tags.filter((t) => t !== tag)
      : [...hold.tags, tag];
    updateHold(activeHoldIndex, { tags: newTags });
  }

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

  const getImageCoords = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.round(
          (e.clientX - rect.left) * (imageDimensions.width / rect.width),
        ),
        y: Math.round(
          (e.clientY - rect.top) * (imageDimensions.height / rect.height),
        ),
      };
    },
    [imageDimensions],
  );

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
    [toFeetCoords, enabledFeatures, lockedUseability, useabilityLocked],
  );

  const getHoldColor = useCallback((u: number, isFoot: boolean) => {
    if (isFoot) {
      const r = Math.round(60 - 60 * u);
      const g = Math.round(0 + 200 * u);
      const b = Math.round(40 + 140 * u);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const r = u < 0.5 ? 255 : Math.round(60 + 195 * (1 - u) * 2);
      const g = u < 0.5 ? Math.round(60 + 160 * u * 2) : 220;
      return `rgb(${r}, ${g}, 60)`;
    }
  }, []);

  const getUseabilityColor = useCallback(
    (u: number) => getHoldColor(u, false),
    [getHoldColor],
  );

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
      if (!enabledFeatures.direction && !enabledFeatures.useability) {
        const idx = addHold(
          x,
          y,
          undefined,
          undefined,
          undefined,
          undefined,
          enabledFeatures.footholds && isAddFoot ? 1 : 0,
          enabledFeatures.tags && stickyTags.length > 0 ? [...stickyTags] : [],
        );
        setActiveHoldIndex(idx);
      } else {
        setAddHoldState({
          isDragging: true,
          holdX: x,
          holdY: y,
          dragX: x,
          dragY: y,
        });
      }
    } else if (mode === "edit") {
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
    } else if (mode === "remove") {
      removeHold(x, y);
    } else if (mode === "select") {
      const hold = findHoldAt(x, y);
      setSelectedHold(hold);
      if (hold) setActiveHoldIndex(hold.hold_index);
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

      const idx = addHold(
        holdX,
        holdY,
        undefined,
        enabledFeatures.direction ? params.pull_x : undefined,
        enabledFeatures.direction ? params.pull_y : undefined,
        enabledFeatures.useability ? params.useability : undefined,
        enabledFeatures.footholds && isAddFoot ? 1 : 0,
        enabledFeatures.tags && stickyTags.length > 0 ? [...stickyTags] : [],
      );
      setActiveHoldIndex(idx);
      setAddHoldState({
        isDragging: false,
        holdX: 0,
        holdY: 0,
        dragX: 0,
        dragY: 0,
      });
    } else if (editHoldState.isDragging && editHoldState.originalHold) {
      const originalHold = editHoldState.originalHold;
      const pixelCoords = toPixelCoords(originalHold);
      const { dragX, dragY } = editHoldState;

      const existingHold = holds.find(
        (h) => h.hold_index === originalHold.hold_index,
      );
      const existingTags = existingHold?.tags ?? originalHold.tags ?? [];
      const params = calculateHoldParams(pixelCoords.x, pixelCoords.y, dragX, dragY);
      const updatedParams: Partial<HoldDetail> = {
        ...(enabledFeatures.direction
          ? { pull_x: params.pull_x, pull_y: params.pull_y }
          : {}),
        ...(enabledFeatures.useability ? { useability: params.useability } : {}),
        ...(enabledFeatures.footholds ? { is_foot: Number(isAddFoot) } : {}),
        ...(enabledFeatures.tags && stickyTags.length > 0
          ? { tags: [...new Set([...existingTags, ...stickyTags])] }
          : {}),
      };

      updateHold(originalHold.hold_index, updatedParams);

      setEditHoldState({
        isDragging: false,
        dragX: 0,
        dragY: 0,
      });
    }
  };

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
    return () => element.removeEventListener("wheel", handleWheel);
  }, []);

  // Hotkeys
  useEffect(() => {
    const handleKeydown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        removeLastHold();
      }
      switch (e.key) {
        case "1":
          e.preventDefault();
          setHoldMode("add");
          break;
        case "2":
          e.preventDefault();
          setHoldMode("edit");
          break;
        case "3":
          e.preventDefault();
          setHoldMode("remove");
          break;
        case "4":
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

  // Tag hotkeys: p=pinch, m=macro, s=sloper, v=versatile, j=jug — sticky toggle
  useEffect(() => {
    const handleKeydown = (e: KeyboardEvent) => {
      if (!enabledFeatures.tags) return;
      const tagMap: Record<string, Tag> = {
        p: "pinch",
        P: "pinch",
        m: "macro",
        M: "macro",
        s: "sloper",
        S: "sloper",
        v: "versatile",
        V: "versatile",
        j: "jug",
        J: "jug",
      };
      const tag = tagMap[e.key];
      if (!tag) return;
      setStickyTags((prev) =>
        prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag],
      );
    };
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [enabledFeatures.tags]);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas?.getContext("2d");
    if (!image || !ctx) return;

    const { width, height } = imageDimensions;
    canvas.width = width || 800;
    canvas.height = height || 600;

    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const isFoot = !!hold.is_foot;
      const scale = height / 1500;
      const sizeMultiplier = isFoot ? 0.5 * scale : scale;
      const useability = hold.useability ?? 0.5;
      const color = getHoldColor(useability, isFoot);

      const circleSize = 4 * sizeMultiplier;
      const arrowSize = 2 * sizeMultiplier;

      if (mode === "select" && selectedHold?.hold_index === hold.hold_index) {
        ctx.beginPath();
        ctx.arc(x, y, 20 * sizeMultiplier, 0, Math.PI * 2);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 4 * sizeMultiplier;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(x, y, circleSize * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = circleSize;
      ctx.stroke();

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
          endY + (arrowSize / 2.0) * Math.sin(angle - Math.PI / 4),
        );
        ctx.lineTo(
          endX - headLength * Math.cos(angle - Math.PI / 4),
          endY - headLength * Math.sin(angle - Math.PI / 4),
        );
        ctx.moveTo(
          endX + (arrowSize / 2.0) * Math.cos(angle + Math.PI / 4),
          endY + (arrowSize / 2.0) * Math.sin(angle + Math.PI / 4),
        );
        ctx.lineTo(
          endX - headLength * Math.cos(angle + Math.PI / 4),
          endY - headLength * Math.sin(angle + Math.PI / 4),
        );
        ctx.stroke();
      }

      ctx.fillStyle = "white";
      ctx.font = `bold ${10 * sizeMultiplier}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(hold.hold_index.toString(), x, y);
    });

    // Add-hold drag preview
    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const params = calculateHoldParams(holdX, holdY, dragX, dragY);
      const sizeMultiplier = isAddFoot ? 0.5 : 1;
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

      if (enabledFeatures.direction) {
        ctx.beginPath();
        ctx.moveTo(dragX, dragY);
        ctx.lineTo(holdX, holdY);
        ctx.strokeStyle = color;
        ctx.lineWidth = arrowSize;
        ctx.stroke();
      }
    }

    // Edit-hold drag preview
    if (editHoldState.isDragging && editHoldState.originalHold) {
      const originalHold = editHoldState.originalHold;
      const pixelCoords = toPixelCoords(originalHold);
      const { dragX, dragY } = editHoldState;
      const params = calculateHoldParams(
        pixelCoords.x,
        pixelCoords.y,
        dragX,
        dragY,
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

  const dragParams = addHoldState.isDragging
    ? {
        ...calculateHoldParams(
          addHoldState.holdX,
          addHoldState.holdY,
          addHoldState.dragX,
          addHoldState.dragY,
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
              editHoldState.dragY,
            );
          })(),
        }
      : { pull_x: 0, pull_y: -1, useability: 0, x: 0, y: 0 };

  const modeButtonStyle = (
    isActive: boolean,
    activeColor: string,
  ): React.CSSProperties => ({
    padding: "6px 10px",
    borderRadius: "var(--radius)",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.6rem",
    fontWeight: 700,
    letterSpacing: "0.08em",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    transition: "all 0.15s",
    border: "none",
    cursor: "pointer",
    background: isActive ? activeColor : "transparent",
    color: isActive ? "#09090b" : "var(--text-muted)",
  });

  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <div
        style={{
          position: "relative",
          height: "calc(100vh - 0px)",
          display: "flex",
          flexDirection: "column",
          background: "var(--bg)",
          overflow: "hidden",
          color: "var(--text-primary)",
        }}
      >
        {/* Feature Menu */}
        {showFeatureMenu && (
          <EnabledFeaturesMenu
            enabledFeatures={enabledFeatures}
            onToggle={handleFeatureToggle}
            onClose={() => setShowFeatureMenu(false)}
          />
        )}

        {/* Header Toolbar */}
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
            zIndex: 10,
          }}
        >
          {/* Left: feature settings */}
          <button
            onClick={() => setShowFeatureMenu(!showFeatureMenu)}
            style={{
              padding: "6px 10px",
              borderRadius: "var(--radius)",
              border: "none",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 0.15s",
              background: showFeatureMenu ? "var(--cyan-dim)" : "transparent",
              color: showFeatureMenu ? "var(--cyan)" : "var(--text-muted)",
            }}
          >
            <Settings size={14} />
            <span
              className="bz-mono"
              style={{
                fontSize: "0.55rem",
                letterSpacing: "0.15em",
                textTransform: "uppercase",
              }}
            >
              Hold Features Settings
            </span>
          </button>

          {/* Center: wall name + mode switcher */}
          <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
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
                style={{
                  fontSize: "0.8rem",
                  color: "var(--text-primary)",
                  letterSpacing: "0.04em",
                }}
              >
                {wall.metadata.name}
              </span>
              <span
                className="bz-mono"
                style={{
                  fontSize: "0.55rem",
                  color: "var(--text-dim)",
                  letterSpacing: "0.1em",
                }}
              >
                / HOLDS
              </span>
            </div>

            {/* Mode switcher */}
            <div
              style={{
                display: "flex",
                background: "var(--bg)",
                borderRadius: "var(--radius)",
                padding: "3px",
                border: "1px solid var(--border)",
                gap: "2px",
              }}
            >
              <button
                onClick={() => setHoldMode("add")}
                style={modeButtonStyle(
                  mode === "add",
                  enabledFeatures.footholds && isAddFoot
                    ? "#9333ea"
                    : "#34d399",
                )}
              >
                {mode === "add" ? (
                  <span
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                    }}
                  >
                    <Plus size={12} />
                    {enabledFeatures.footholds && isAddFoot ? "FOOT" : "HAND"}
                  </span>
                ) : (
                  <Plus size={13} />
                )}
              </button>
              <button
                onClick={() => setHoldMode("edit")}
                style={modeButtonStyle(mode === "edit", "#f59e0b")}
              >
                <Edit size={13} />
              </button>
              <button
                onClick={() => setHoldMode("remove")}
                style={modeButtonStyle(mode === "remove", "#ef4444")}
              >
                <Eraser size={13} />
              </button>
              <button
                onClick={() => setHoldMode("select")}
                style={modeButtonStyle(mode === "select", "#3b82f6")}
              >
                <Hand size={13} />
              </button>
            </div>
          </div>

          {/* Right: actions */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              onClick={() => {
                clearHolds();
                setSelectedHold(null);
              }}
              className="bz-mono"
              style={{
                padding: "6px 12px",
                borderRadius: "var(--radius)",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(239,68,68,0.1)";
                e.currentTarget.style.color = "#ef4444";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.color = "var(--text-muted)";
              }}
            >
              CLEAR
            </button>
            <button
              onClick={() =>
                navigate({ to: "/$wallId/set", params: { wallId } })
              }
              className="bz-mono"
              style={{
                padding: "6px 12px",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.6rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                transition: "color 0.15s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.color = "var(--text-primary)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.color = "var(--text-muted)")
              }
            >
              CANCEL
            </button>
            <button
              onClick={async () => {
                setIsSubmitting(true);
                setError(null);
                try {
                  await setHolds(wallId, holds);
                  navigate({ to: "/$wallId/set", params: { wallId } });
                } catch (err) {
                  setError(
                    err instanceof Error ? err.message : "Failed to save holds",
                  );
                } finally {
                  setIsSubmitting(false);
                }
              }}
              disabled={isSubmitting}
              className="bz-oswald"
              style={{
                padding: "6px 18px",
                background: isSubmitting ? "var(--surface2)" : "var(--cyan)",
                color: isSubmitting ? "var(--text-muted)" : "#09090b",
                border: "none",
                borderRadius: "var(--radius)",
                fontFamily: "'Oswald', sans-serif",
                fontSize: "0.75rem",
                letterSpacing: "0.1em",
                fontWeight: 700,
                textTransform: "uppercase",
                cursor: isSubmitting ? "not-allowed" : "pointer",
                transition: "all 0.15s",
                opacity: isSubmitting ? 0.6 : 1,
              }}
            >
              {isSubmitting ? "SAVING…" : "SAVE HOLDS"}
            </button>
          </div>
        </header>

        {error && (
          <div
            className="bz-mono"
            style={{
              padding: "8px 20px",
              background: "rgba(248,113,113,0.08)",
              borderBottom: "1px solid rgba(248,113,113,0.2)",
              fontSize: "0.65rem",
              color: "#f87171",
            }}
          >
            {error}
          </div>
        )}

        {/* Main content */}
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          {/* Canvas area */}
          <div
            ref={wrapperRef}
            style={{
              flex: 1,
              overflow: "hidden",
              background: "var(--bg)",
              cursor: "crosshair",
            }}
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
            activeHold={activeHold}
            onTagToggle={handleTagToggle}
            stickyTags={stickyTags}
            onStickyTagToggle={(tag) =>
              setStickyTags((prev) =>
                prev.includes(tag)
                  ? prev.filter((t) => t !== tag)
                  : [...prev, tag],
              )
            }
          />
        </div>
      </div>
    </>
  );
}
