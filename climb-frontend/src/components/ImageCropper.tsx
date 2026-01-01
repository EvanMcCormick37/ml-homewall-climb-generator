import { useRef, useCallback } from "react";
import type { CropArea } from "@/hooks";

export interface ImageCropperProps {
  imageUrl: string;
  cropArea: CropArea;
  isDragging: boolean;
  onStartDrag: (mode: "move" | "resize", handle?: string) => void;
  onUpdateDrag: (
    clientX: number,
    clientY: number,
    containerRect: DOMRect
  ) => void;
  onEndDrag: () => void;
}

export default function ImageCropper({
  imageUrl,
  cropArea,
  isDragging,
  onStartDrag,
  onUpdateDrag,
  onEndDrag,
}: ImageCropperProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      onUpdateDrag(e.clientX, e.clientY, rect);
    },
    [isDragging, onUpdateDrag]
  );

  const handleMouseUp = useCallback(() => {
    onEndDrag();
  }, [onEndDrag]);

  const handleMouseLeave = useCallback(() => {
    if (isDragging) onEndDrag();
  }, [isDragging, onEndDrag]);

  // Resize handles
  const handles = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

  const getHandleStyle = (handle: string): React.CSSProperties => {
    const size = 12;
    const base: React.CSSProperties = {
      position: "absolute",
      width: size,
      height: size,
      backgroundColor: "#fff",
      border: "2px solid #a855f7",
      borderRadius: 2,
      cursor: `${handle}-resize`,
      zIndex: 10,
    };

    const halfSize = size / 2;

    switch (handle) {
      case "nw":
        return { ...base, top: -halfSize, left: -halfSize };
      case "n":
        return {
          ...base,
          top: -halfSize,
          left: "50%",
          transform: "translateX(-50%)",
        };
      case "ne":
        return { ...base, top: -halfSize, right: -halfSize };
      case "e":
        return {
          ...base,
          top: "50%",
          right: -halfSize,
          transform: "translateY(-50%)",
        };
      case "se":
        return { ...base, bottom: -halfSize, right: -halfSize };
      case "s":
        return {
          ...base,
          bottom: -halfSize,
          left: "50%",
          transform: "translateX(-50%)",
        };
      case "sw":
        return { ...base, bottom: -halfSize, left: -halfSize };
      case "w":
        return {
          ...base,
          top: "50%",
          left: -halfSize,
          transform: "translateY(-50%)",
        };
      default:
        return base;
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full select-none overflow-hidden rounded-lg bg-zinc-900"
      style={{ aspectRatio: "auto" }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
    >
      {/* Base image */}
      <img
        src={imageUrl}
        alt="Wall to crop"
        className="block w-full h-auto"
        draggable={false}
      />

      {/* Darkened overlay outside crop area */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `linear-gradient(to right, 
            rgba(0,0,0,0.7) ${cropArea.x * 100}%, 
            transparent ${cropArea.x * 100}%, 
            transparent ${(cropArea.x + cropArea.width) * 100}%, 
            rgba(0,0,0,0.7) ${(cropArea.x + cropArea.width) * 100}%
          )`,
        }}
      />
      <div
        className="absolute pointer-events-none"
        style={{
          left: `${cropArea.x * 100}%`,
          right: `${(1 - cropArea.x - cropArea.width) * 100}%`,
          top: 0,
          height: `${cropArea.y * 100}%`,
          backgroundColor: "rgba(0,0,0,0.7)",
        }}
      />
      <div
        className="absolute pointer-events-none"
        style={{
          left: `${cropArea.x * 100}%`,
          right: `${(1 - cropArea.x - cropArea.width) * 100}%`,
          bottom: 0,
          height: `${(1 - cropArea.y - cropArea.height) * 100}%`,
          backgroundColor: "rgba(0,0,0,0.7)",
        }}
      />

      {/* Crop area with border */}
      <div
        className="absolute cursor-move"
        style={{
          left: `${cropArea.x * 100}%`,
          top: `${cropArea.y * 100}%`,
          width: `${cropArea.width * 100}%`,
          height: `${cropArea.height * 100}%`,
          border: "1px dashed #a855f7",
          boxShadow: "0 0 0 9999px transparent",
        }}
        onMouseDown={(e) => {
          e.preventDefault();
          onStartDrag("move");
        }}
      >
        {/* Grid lines */}
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute w-full h-px bg-purple-400/30"
            style={{ top: "33.33%" }}
          />
          <div
            className="absolute w-full h-px bg-purple-400/30"
            style={{ top: "66.66%" }}
          />
          <div
            className="absolute h-full w-px bg-purple-400/30"
            style={{ left: "33.33%" }}
          />
          <div
            className="absolute h-full w-px bg-purple-400/30"
            style={{ left: "66.66%" }}
          />
        </div>

        {/* Resize handles */}
        {handles.map((handle) => (
          <div
            key={handle}
            style={getHandleStyle(handle)}
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onStartDrag("resize", handle);
            }}
          />
        ))}
      </div>

      {/* Instructions */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-black/70 rounded text-xs text-zinc-300">
        Drag to move • Drag corners to resize
      </div>
    </div>
  );
}
