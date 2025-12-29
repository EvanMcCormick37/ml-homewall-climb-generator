import { useState, useCallback } from "react";
import type { HoldDetail } from "@/types";

interface Dimensions {
  width: number;
  height: number;
}

export function useHolds(
  imageDimensions: Dimensions,
  wallDimensions: Dimensions
) {
  const [holds, setHolds] = useState<HoldDetail[]>([]);

  // Convert pixel coordinates to feet
  const toFeetCoords = useCallback(
    (pixelX: number, pixelY: number) => {
      const xFeet = (pixelX / imageDimensions.width) * wallDimensions.width;
      const yFeet =
        ((imageDimensions.height - pixelY) / imageDimensions.height) *
        wallDimensions.height;
      return { x: xFeet, y: yFeet };
    },
    [imageDimensions, wallDimensions]
  );

  // Convert feet coordinates to pixels
  const toPixelCoords = useCallback(
    (hold: HoldDetail) => {
      const pixelX = (hold.x / wallDimensions.width) * imageDimensions.width;
      const pixelY =
        imageDimensions.height -
        (hold.y / wallDimensions.height) * imageDimensions.height;
      return { x: pixelX, y: pixelY };
    },
    [imageDimensions, wallDimensions]
  );

  // Add a new hold with optional features
  const addHold = useCallback(
    (
      pixelX: number,
      pixelY: number,
      pull_x?: number,
      pull_y?: number,
      useability?: number,
      is_foot?: number
    ) => {
      const { x, y } = toFeetCoords(pixelX, pixelY);
      const newHold: HoldDetail = {
        hold_index: holds.length,
        x,
        y,
        pull_x: pull_x ?? null,
        pull_y: pull_y ?? null,
        useability: useability ?? null,
        is_foot: is_foot ?? 0,
      };

      setHolds((prev) => [...prev, newHold]);
    },
    [holds.length, toFeetCoords]
  );

  // Remove a hold at pixel coordinates (finds closest)
  const removeHold = useCallback(
    (pixelX: number, pixelY: number) => {
      let closestIndex = -1;
      let minDist = Infinity;

      holds.forEach((hold, index) => {
        const { x, y } = toPixelCoords(hold);
        const dist = Math.sqrt((x - pixelX) ** 2 + (y - pixelY) ** 2);
        if (dist < minDist && dist < 30) {
          minDist = dist;
          closestIndex = index;
        }
      });

      if (closestIndex !== -1) {
        setHolds((prev) => {
          const updated = prev.filter((_, i) => i !== closestIndex);
          // Re-index remaining holds
          return updated.map((hold, i) => ({
            ...hold,
            hold_index: i,
          }));
        });
      }
    },
    [holds, toPixelCoords]
  );

  // Remove hold by index
  const removeHoldByIndex = useCallback((holdIndex: number) => {
    setHolds((prev) => {
      const updated = prev.filter((h) => h.hold_index !== holdIndex);
      // Re-index remaining holds
      return updated.map((hold, i) => ({
        ...hold,
        hold_index: i,
      }));
    });
  }, []);

  // Remove the last added hold
  const removeLastHold = useCallback(() => {
    setHolds((prev) => {
      if (prev.length === 0) return prev;
      return prev.slice(0, -1);
    });
  }, []);

  // Find hold at pixel coordinates
  const findHoldAt = useCallback(
    (pixelX: number, pixelY: number): HoldDetail | null => {
      let closestHold: HoldDetail | null = null;
      let minDist = Infinity;

      holds.forEach((hold) => {
        const { x, y } = toPixelCoords(hold);
        const dist = Math.sqrt((x - pixelX) ** 2 + (y - pixelY) ** 2);
        if (dist < minDist && dist < 30) {
          minDist = dist;
          closestHold = hold;
        }
      });

      return closestHold;
    },
    [holds, toPixelCoords]
  );

  // Clear all holds
  const clearHolds = useCallback(() => {
    setHolds([]);
  }, []);

  // Load holds from API data
  const loadHolds = useCallback((holdsData: HoldDetail[]) => {
    setHolds(holdsData);
  }, []);

  return {
    holds,
    addHold,
    removeHold,
    removeHoldByIndex,
    removeLastHold,
    findHoldAt,
    clearHolds,
    loadHolds,
    toPixelCoords,
    toFeetCoords,
  };
}
