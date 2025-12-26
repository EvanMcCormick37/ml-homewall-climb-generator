import { useState, useCallback, useRef } from "react";

export interface CropArea {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface UseImageCropReturn {
  imageUrl: string | null;
  cropArea: CropArea | null;
  isDragging: boolean;
  dragMode: "move" | "resize" | null;
  resizeHandle: string | null;
  setImage: (file: File) => void;
  setCropArea: (area: CropArea) => void;
  startDrag: (mode: "move" | "resize", handle?: string) => void;
  updateDrag: (
    clientX: number,
    clientY: number,
    containerRect: DOMRect
  ) => void;
  endDrag: () => void;
  resetCrop: () => void;
  getCroppedImage: () => Promise<Blob | null>;
}

export function useImageCrop(): UseImageCropReturn {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [cropArea, setCropArea] = useState<CropArea | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragMode, setDragMode] = useState<"move" | "resize" | null>(null);
  const [resizeHandle, setResizeHandle] = useState<string | null>(null);
  const dragStart = useRef<{ x: number; y: number; area: CropArea } | null>(
    null
  );
  const imageDimensions = useRef<{ width: number; height: number } | null>(
    null
  );

  const setImage = useCallback(
    (file: File) => {
      // Revoke previous URL if exists
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }

      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setImageFile(file);

      // Load image to get dimensions and set initial crop area
      const img = new Image();
      img.onload = () => {
        imageDimensions.current = { width: img.width, height: img.height };
        // Initial crop area covers 80% centered
        const margin = 0.1;
        setCropArea({
          x: margin,
          y: margin,
          width: 1 - 2 * margin,
          height: 1 - 2 * margin,
        });
      };
      img.src = url;
    },
    [imageUrl]
  );

  const startDrag = useCallback((mode: "move" | "resize", handle?: string) => {
    setIsDragging(true);
    setDragMode(mode);
    if (handle) setResizeHandle(handle);
  }, []);

  const updateDrag = useCallback(
    (clientX: number, clientY: number, containerRect: DOMRect) => {
      if (!isDragging || !cropArea || !dragMode) return;

      // Convert client coordinates to normalized (0-1) coordinates
      const normX = (clientX - containerRect.left) / containerRect.width;
      const normY = (clientY - containerRect.top) / containerRect.height;

      if (!dragStart.current) {
        dragStart.current = { x: normX, y: normY, area: { ...cropArea } };
        return;
      }

      const deltaX = normX - dragStart.current.x;
      const deltaY = normY - dragStart.current.y;
      const startArea = dragStart.current.area;

      if (dragMode === "move") {
        // Move the crop area
        let newX = startArea.x + deltaX;
        let newY = startArea.y + deltaY;

        // Clamp to bounds
        newX = Math.max(0, Math.min(1 - startArea.width, newX));
        newY = Math.max(0, Math.min(1 - startArea.height, newY));

        setCropArea({ ...startArea, x: newX, y: newY });
      } else if (dragMode === "resize" && resizeHandle) {
        // Resize the crop area
        let newArea = { ...startArea };
        const minSize = 0.1;

        switch (resizeHandle) {
          case "nw":
            newArea.x = Math.min(
              startArea.x + startArea.width - minSize,
              startArea.x + deltaX
            );
            newArea.y = Math.min(
              startArea.y + startArea.height - minSize,
              startArea.y + deltaY
            );
            newArea.width = startArea.width - (newArea.x - startArea.x);
            newArea.height = startArea.height - (newArea.y - startArea.y);
            break;
          case "ne":
            newArea.y = Math.min(
              startArea.y + startArea.height - minSize,
              startArea.y + deltaY
            );
            newArea.width = Math.max(minSize, startArea.width + deltaX);
            newArea.height = startArea.height - (newArea.y - startArea.y);
            break;
          case "sw":
            newArea.x = Math.min(
              startArea.x + startArea.width - minSize,
              startArea.x + deltaX
            );
            newArea.width = startArea.width - (newArea.x - startArea.x);
            newArea.height = Math.max(minSize, startArea.height + deltaY);
            break;
          case "se":
            newArea.width = Math.max(minSize, startArea.width + deltaX);
            newArea.height = Math.max(minSize, startArea.height + deltaY);
            break;
          case "n":
            newArea.y = Math.min(
              startArea.y + startArea.height - minSize,
              startArea.y + deltaY
            );
            newArea.height = startArea.height - (newArea.y - startArea.y);
            break;
          case "s":
            newArea.height = Math.max(minSize, startArea.height + deltaY);
            break;
          case "e":
            newArea.width = Math.max(minSize, startArea.width + deltaX);
            break;
          case "w":
            newArea.x = Math.min(
              startArea.x + startArea.width - minSize,
              startArea.x + deltaX
            );
            newArea.width = startArea.width - (newArea.x - startArea.x);
            break;
        }

        // Clamp to bounds
        newArea.x = Math.max(0, newArea.x);
        newArea.y = Math.max(0, newArea.y);
        newArea.width = Math.min(1 - newArea.x, newArea.width);
        newArea.height = Math.min(1 - newArea.y, newArea.height);

        setCropArea(newArea);
      }
    },
    [isDragging, cropArea, dragMode, resizeHandle]
  );

  const endDrag = useCallback(() => {
    setIsDragging(false);
    setDragMode(null);
    setResizeHandle(null);
    dragStart.current = null;
  }, []);

  const resetCrop = useCallback(() => {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setImageUrl(null);
    setImageFile(null);
    setCropArea(null);
    imageDimensions.current = null;
  }, [imageUrl]);

  const getCroppedImage = useCallback(async (): Promise<Blob | null> => {
    if (!imageFile || !cropArea || !imageDimensions.current) return null;

    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(null);
          return;
        }

        // Calculate pixel coordinates
        const sx = cropArea.x * img.width;
        const sy = cropArea.y * img.height;
        const sWidth = cropArea.width * img.width;
        const sHeight = cropArea.height * img.height;

        canvas.width = sWidth;
        canvas.height = sHeight;

        ctx.drawImage(img, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight);

        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.9);
      };
      img.src = URL.createObjectURL(imageFile);
    });
  }, [imageFile, cropArea]);

  return {
    imageUrl,
    cropArea,
    isDragging,
    dragMode,
    resizeHandle,
    setImage,
    setCropArea,
    startDrag,
    updateDrag,
    endDrag,
    resetCrop,
    getCroppedImage,
  };
}
