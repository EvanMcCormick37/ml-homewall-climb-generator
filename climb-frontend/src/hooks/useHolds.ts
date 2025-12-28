import { useState, useCallback } from "react";
import type { HoldDetail } from "@/types";

interface ImageDimensions {
  width: number;
  height: number;
}

interface WallDimensions {
  width: number; // in feet
  height: number; // in feet
}

interface UseHoldsReturn {
  holds: HoldDetail[];
  addHold: (
    pixelX: number,
    pixelY: number,
    pull_x: number,
    pull_y: number,
    useability: number
  ) => void;
  removeHold: (pixelX: number, pixelY: number, radius?: number) => void;
  removeHoldByIndex: (holdIndex: number) => void;
  removeLastHold: () => HoldDetail | null;
  findHoldAt: (
    pixelX: number,
    pixelY: number,
    radius?: number
  ) => HoldDetail | null;
  clearHolds: () => void;
  loadHolds: (holds: HoldDetail[]) => void;
  // Coordinate conversion utilities
  toPixelCoords: (hold: HoldDetail) => { x: number; y: number };
  toFeetCoords: (pixelX: number, pixelY: number) => { x: number; y: number };
}

/**
 * Custom hook for managing hold annotation and operations.
 *
 * Holds are stored with x/y in feet. This hook provides utilities to convert
 * between feet (for storage/API) and pixels (for canvas rendering).
 *
 * @param imageDimensions - The dimensions of the wall image in pixels
 * @param wallDimensions - The dimensions of the wall in feet
 * @returns Hold state and CRUD operations
 */
export function useHolds(
  imageDimensions: ImageDimensions,
  wallDimensions: WallDimensions
): UseHoldsReturn {
  const [holds, setHolds] = useState<HoldDetail[]>([]);
  const [nextIndex, setNextIndex] = useState(0);

  // Convert from feet to pixel coordinates
  const toPixelCoords = useCallback(
    (hold: HoldDetail): { x: number; y: number } => {
      const { width: imgW, height: imgH } = imageDimensions;
      const { width: wallW, height: wallH } = wallDimensions;

      return {
        x: (hold.x / wallW) * imgW,
        y: (1 - hold.y / wallH) * imgH, // y is inverted (0 at bottom in feet, 0 at top in pixels)
      };
    },
    [imageDimensions, wallDimensions]
  );

  // Convert from pixel to feet coordinates
  const toFeetCoords = useCallback(
    (pixelX: number, pixelY: number): { x: number; y: number } => {
      const { width: imgW, height: imgH } = imageDimensions;
      const { width: wallW, height: wallH } = wallDimensions;

      return {
        x: (pixelX / imgW) * wallW,
        y: (1 - pixelY / imgH) * wallH, // invert y back to feet
      };
    },
    [imageDimensions, wallDimensions]
  );

  // Load holds from data (e.g., from API)
  const loadHolds = useCallback((holdsData: HoldDetail[]) => {
    setHolds(holdsData || []);
    const maxIndex = Math.max(
      ...(holdsData || []).map((h) => h.hold_index),
      -1
    );
    setNextIndex(maxIndex + 1);
  }, []);

  // Add a new hold at pixel position with pull direction and useability
  const addHold = useCallback(
    (
      pixelX: number,
      pixelY: number,
      pull_x: number,
      pull_y: number,
      useability: number
    ) => {
      const { width: imgW, height: imgH } = imageDimensions;
      const { width: wallW, height: wallH } = wallDimensions;
      if (!imgW || !imgH || !wallW || !wallH) return;

      const feetCoords = toFeetCoords(pixelX, pixelY);

      const newHold: HoldDetail = {
        hold_index: nextIndex,
        x: feetCoords.x,
        y: feetCoords.y,
        pull_x: pull_x,
        pull_y: pull_y,
        useability: useability,
      };

      setHolds((prev) => [...prev, newHold]);
      setNextIndex((prev) => prev + 1);
    },
    [nextIndex, imageDimensions, wallDimensions, toFeetCoords]
  );

  // Remove hold nearest to pixel position
  const removeHold = useCallback(
    (pixelX: number, pixelY: number, radius: number = 40) => {
      setHolds((prev) => {
        let minDist = Infinity;
        let nearestIdx = -1;

        prev.forEach((hold, idx) => {
          const pixelCoords = toPixelCoords(hold);
          const dist = Math.sqrt(
            (pixelCoords.x - pixelX) ** 2 + (pixelCoords.y - pixelY) ** 2
          );

          if (dist < minDist && dist < radius) {
            minDist = dist;
            nearestIdx = idx;
          }
        });

        if (nearestIdx >= 0) {
          return prev.filter((_, idx) => idx !== nearestIdx);
        }
        return prev;
      });
    },
    [toPixelCoords]
  );

  // Remove hold by index
  const removeHoldByIndex = useCallback((holdIndex: number) => {
    setHolds((prev) => prev.filter((hold) => hold.hold_index !== holdIndex));
  }, []);

  // Remove the most recently added hold and return it
  const removeLastHold = useCallback((): HoldDetail | null => {
    let removedHold: HoldDetail | null = null;
    setHolds((prev) => {
      if (prev.length === 0) return prev;
      removedHold = prev[prev.length - 1];
      return prev.slice(0, -1);
    });
    return removedHold;
  }, []);

  // Find hold at pixel position
  const findHoldAt = useCallback(
    (
      pixelX: number,
      pixelY: number,
      radius: number = 40
    ): HoldDetail | null => {
      let nearest: HoldDetail | null = null;
      let minDist = Infinity;

      for (const hold of holds) {
        const pixelCoords = toPixelCoords(hold);
        const dist = Math.sqrt(
          (pixelCoords.x - pixelX) ** 2 + (pixelCoords.y - pixelY) ** 2
        );

        if (dist < minDist && dist < radius) {
          minDist = dist;
          nearest = hold;
        }
      }
      return nearest;
    },
    [holds, toPixelCoords]
  );

  // Clear all holds
  const clearHolds = useCallback(() => {
    setHolds([]);
    setNextIndex(0);
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
