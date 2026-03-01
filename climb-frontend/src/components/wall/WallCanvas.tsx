import { useState, useCallback, useRef, useEffect } from "react";
import { getWallPhotoUrl } from "@/api/walls";
import type { HoldDetail, Holdset } from "@/types";
import {
  HOLD_STROKE_COLOR,
  DEFAULT_DISPLAY_SETTINGS,
  type DisplaySettings,
} from "./types";

// ─── WallCanvas ──────────────────────────────────────────────────────────────

export interface WallCanvasProps {
  wallId: string;
  holds: HoldDetail[];
  wallDimensions: { width: number; height: number };
  selectedHoldset: Holdset | null;
  imageDimensions: { width: number; height: number };
  onImageLoad: (d: { width: number; height: number }) => void;
  /** Display settings for hold rendering. Falls back to defaults. */
  displaySettings?: DisplaySettings;
  /** Called when a hold is clicked. If omitted, holds are not interactive. */
  onHoldClick?: (holdIndex: number) => void;
  /** Swipe callbacks for mobile climb navigation. */
  onSwipeNext?: () => void;
  onSwipePrev?: () => void;
}

export function WallCanvas({
  wallId,
  holds,
  wallDimensions,
  selectedHoldset,
  imageDimensions,
  onImageLoad,
  displaySettings = DEFAULT_DISPLAY_SETTINGS,
  onHoldClick,
  onSwipeNext,
  onSwipePrev,
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
  const touchRef = useRef({
    lastTouchX: 0,
    lastTouchY: 0,
    lastDist: 0,
    isTwoFinger: false,
    startX: 0,
    startY: 0,
    startTime: 0,
    startViewX: 0,
    startViewY: 0,
    moved: false,
  });
  const [swipeHint, setSwipeHint] = useState<"left" | "right" | null>(null);

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

  const toPixelCoords = useCallback(
    (hold: HoldDetail) => ({
      x: (hold.x / wallDimensions.width) * imageDimensions.width,
      y: (1 - hold.y / wallDimensions.height) * imageDimensions.height,
    }),
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
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

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

    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const baseScale = height / 500;
      const radius = 10 * baseScale * userScale;
      const isUsed = usedHolds.has(hold.hold_index);
      const isStart = startHolds.has(hold.hold_index),
        isFinish = finishHolds.has(hold.hold_index);
      const isHand = handHolds.has(hold.hold_index),
        isFoot = footHolds.has(hold.hold_index);
      const baseAlpha = selectedHoldset ? (isUsed ? 1 : 0.2) : 0.5;
      const alpha = isUsed ? baseAlpha * userOpacity : baseAlpha;
      let strokeColor = HOLD_STROKE_COLOR;
      if (selectedHoldset && isUsed) {
        if (colorMode === "uniform") strokeColor = uniformColor;
        else if (isStart) strokeColor = displaySettings.categoryColors.start;
        else if (isFinish) strokeColor = displaySettings.categoryColors.finish;
        else if (isHand) strokeColor = displaySettings.categoryColors.hand;
        else if (isFoot) strokeColor = displaySettings.categoryColors.foot;
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
  }, [image, imageDimensions, holds, selectedHoldset, toPixelCoords, displaySettings]);

  // ─── Coordinate helpers ──────────────────────────────────────────────────────

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

  const findHoldAt = useCallback(
    (pixelX: number, pixelY: number) => {
      for (const hold of holds) {
        const { x, y } = toPixelCoords(hold);
        if (Math.sqrt((x - pixelX) ** 2 + (y - pixelY) ** 2) < 25) return hold;
      }
      return null;
    },
    [holds, toPixelCoords],
  );

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (!onHoldClick) return;
      const { x, y } = getImageCoords(e);
      const hold = findHoldAt(x, y);
      if (hold) onHoldClick(hold.hold_index);
    },
    [getImageCoords, findHoldAt, onHoldClick],
  );

  // ─── Mouse pan/zoom ────────────────────────────────────────────────────────

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
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3)
      panDragRef.current.isDragging = true;
    if (panDragRef.current.isDragging)
      setViewTransform((prev) => ({
        ...prev,
        x: panDragRef.current.startViewX + dx,
        y: panDragRef.current.startViewY + dy,
      }));
  }, []);

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!panDragRef.current.isDragging) handleCanvasClick(e);
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
      const mouseX = e.clientX - rect.left,
        mouseY = e.clientY - rect.top;
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

  // ─── Touch gestures ────────────────────────────────────────────────────────

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;
    const getTouchDist = (t: TouchList) =>
      Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const getTouchMid = (t: TouchList) => ({
      x: (t[0].clientX + t[1].clientX) / 2,
      y: (t[0].clientY + t[1].clientY) / 2,
    });
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        touchRef.current = {
          ...touchRef.current,
          lastTouchX: e.touches[0].clientX,
          lastTouchY: e.touches[0].clientY,
          startX: e.touches[0].clientX,
          startY: e.touches[0].clientY,
          startTime: Date.now(),
          isTwoFinger: false,
          moved: false,
        };
        setViewTransform((prev) => {
          touchRef.current.startViewX = prev.x;
          touchRef.current.startViewY = prev.y;
          return prev;
        });
      } else if (e.touches.length === 2) {
        e.preventDefault();
        touchRef.current.isTwoFinger = true;
        touchRef.current.lastDist = getTouchDist(e.touches);
        const mid = getTouchMid(e.touches);
        touchRef.current.lastTouchX = mid.x;
        touchRef.current.lastTouchY = mid.y;
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        e.preventDefault();
        const rect = element.getBoundingClientRect();
        const mid = getTouchMid(e.touches);
        const dist = getTouchDist(e.touches);
        const pinchFactor = dist / (touchRef.current.lastDist || dist);
        const dx = mid.x - touchRef.current.lastTouchX,
          dy = mid.y - touchRef.current.lastTouchY;
        const midX = mid.x - rect.left,
          midY = mid.y - rect.top;
        setViewTransform((prev) => {
          const newZoom = Math.max(0.1, Math.min(10, prev.zoom * pinchFactor));
          const scale = newZoom / prev.zoom;
          return {
            zoom: newZoom,
            x: midX - (midX - prev.x) * scale + dx,
            y: midY - (midY - prev.y) * scale + dy,
          };
        });
        touchRef.current.lastDist = dist;
        touchRef.current.lastTouchX = mid.x;
        touchRef.current.lastTouchY = mid.y;
      } else if (e.touches.length === 1 && !touchRef.current.isTwoFinger) {
        e.preventDefault();
        const dx = e.touches[0].clientX - touchRef.current.startX;
        const dy = e.touches[0].clientY - touchRef.current.startY;
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) touchRef.current.moved = true;
        if (touchRef.current.moved)
          setViewTransform((prev) => ({
            ...prev,
            x: touchRef.current.startViewX + dx,
            y: touchRef.current.startViewY + dy,
          }));
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2) touchRef.current.isTwoFinger = false;
      if (e.changedTouches.length === 1 && !touchRef.current.isTwoFinger) {
        const dx = e.changedTouches[0].clientX - touchRef.current.startX;
        const dy = e.changedTouches[0].clientY - touchRef.current.startY;
        const dt = Date.now() - touchRef.current.startTime;
        if (
          Math.abs(dx) > 60 &&
          Math.abs(dy) < 80 &&
          dt < 400 &&
          Math.abs(dx) > Math.abs(dy) * 1.5
        ) {
          if (dx < 0 && onSwipeNext) {
            setSwipeHint("left");
            setTimeout(() => setSwipeHint(null), 600);
            onSwipeNext();
          } else if (dx > 0 && onSwipePrev) {
            setSwipeHint("right");
            setTimeout(() => setSwipeHint(null), 600);
            onSwipePrev();
          }
        }
      }
    };
    element.addEventListener("touchstart", onTouchStart, { passive: false });
    element.addEventListener("touchmove", onTouchMove, { passive: false });
    element.addEventListener("touchend", onTouchEnd, { passive: true });
    return () => {
      element.removeEventListener("touchstart", onTouchStart);
      element.removeEventListener("touchmove", onTouchMove);
      element.removeEventListener("touchend", onTouchEnd);
    };
  }, [onSwipeNext, onSwipePrev]);

  // ─── Render ────────────────────────────────────────────────────────────────

  const { zoom, x, y } = viewTransform;
  const { width, height } = imageDimensions;

  return (
    <div
      ref={wrapperRef}
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "var(--bg)",
        position: "relative",
        cursor: onHoldClick ? "crosshair" : "grab",
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        panDragRef.current.isDragging = false;
      }}
    >
      <div style={{ transform: `translate(${x}px, ${y}px)` }}>
        <canvas
          ref={canvasRef}
          style={{
            width: (width || 800) * zoom,
            height: (height || 600) * zoom,
          }}
        />
      </div>

      {swipeHint && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
          }}
        >
          <div
            className="bz-mono"
            style={{
              background: "rgba(0,0,0,0.7)",
              borderRadius: "var(--radius)",
              padding: "10px 20px",
              color: "var(--cyan)",
              fontSize: "0.7rem",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              border: "1px solid var(--cyan)",
            }}
          >
            {swipeHint === "left" ? "→ Next" : "← Prev"}
          </div>
        </div>
      )}
    </div>
  );
}
