import { useState, useCallback } from "react";
import type { HoldDetail } from "@/types";

interface ImageDimensions {
  width: number;
  height: number;
}

interface HoldWithPixels extends HoldDetail {
  pixel_x?: number;
  pixel_y?: number;
}

interface UseHoldsReturn {
  holds: HoldWithPixels[];
  addHold: (
    x: number,
    y: number,
    pull_x: number,
    pull_y: number,
    useability: number
  ) => void;
  removeHold: (x: number, y: number, radius?: number) => void;
  removeHoldById: (holdId: number) => void;
  removeLastHold: () => HoldWithPixels | null;
  findHoldAt: (x: number, y: number, radius?: number) => HoldWithPixels | null;
  clearHolds: () => void;
  loadHolds: (data: { holds: HoldDetail[] }) => void;
}

/**
 * Custom hook for managing hold annotation and operations.
 * Simplified version for the climb-front-end, focused on hold editing.
 *
 * @param imageDimensions - The dimensions of the wall image
 * @returns Hold state and CRUD operations
 */
export function useHolds(imageDimensions: ImageDimensions): UseHoldsReturn {
  const [holds, setHolds] = useState<HoldWithPixels[]>([]);
  const [nextId, setNextId] = useState(0);

  // Load holds from data (e.g., from API)
  const loadHolds = useCallback((data: { holds: HoldDetail[] }) => {
    const loadedHolds = data.holds || [];
    // Convert normalized coordinates to pixel coordinates
    const holdsWithPixels: HoldWithPixels[] = loadedHolds.map((hold) => ({
      ...hold,
      pixel_x: hold.norm_x * imageDimensions.width,
      pixel_y: (1 - hold.norm_y) * imageDimensions.height,
    }));
    setHolds(holdsWithPixels);
    const maxId = Math.max(...loadedHolds.map((h) => h.hold_id), -1);
    setNextId(maxId + 1);
  }, [imageDimensions.width, imageDimensions.height]);

  // Add a new hold at pixel position with pull direction and useability
  const addHold = useCallback(
    (
      x: number,
      y: number,
      pull_x: number,
      pull_y: number,
      useability: number
    ) => {
      const { width, height } = imageDimensions;
      if (!width || !height) return;

      const newHold: HoldWithPixels = {
        hold_id: nextId,
        pixel_x: x,
        pixel_y: y,
        norm_x: x / width,
        norm_y: 1 - y / height,
        pull_x: pull_x,
        pull_y: pull_y,
        useability: useability,
      };

      setHolds((prev) => [...prev, newHold]);
      setNextId((prev) => prev + 1);
    },
    [nextId, imageDimensions]
  );

  // Remove hold nearest to pixel position
  const removeHold = useCallback((x: number, y: number, radius: number = 40) => {
    setHolds((prev) => {
      let minDist = Infinity;
      let nearestIdx = -1;

      prev.forEach((hold, idx) => {
        const hx = hold.pixel_x ?? hold.norm_x * imageDimensions.width;
        const hy = hold.pixel_y ?? (1 - hold.norm_y) * imageDimensions.height;
        const dist = Math.sqrt((hx - x) ** 2 + (hy - y) ** 2);

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
  }, [imageDimensions]);

  // Remove hold by ID
  const removeHoldById = useCallback((holdId: number) => {
    setHolds((prev) => prev.filter((hold) => hold.hold_id !== holdId));
  }, []);

  // Remove the most recently added hold and return it
  const removeLastHold = useCallback((): HoldWithPixels | null => {
    let removedHold: HoldWithPixels | null = null;
    setHolds((prev) => {
      if (prev.length === 0) return prev;
      removedHold = prev[prev.length - 1];
      return prev.slice(0, -1);
    });
    return removedHold;
  }, []);

  // Find hold at pixel position
  const findHoldAt = useCallback(
    (x: number, y: number, radius: number = 40): HoldWithPixels | null => {
      let nearest: HoldWithPixels | null = null;
      let minDist = Infinity;

      for (const hold of holds) {
        const hx = hold.pixel_x ?? hold.norm_x * imageDimensions.width;
        const hy = hold.pixel_y ?? (1 - hold.norm_y) * imageDimensions.height;
        const dist = Math.sqrt((hx - x) ** 2 + (hy - y) ** 2);

        if (dist < minDist && dist < radius) {
          minDist = dist;
          nearest = hold;
        }
      }
      return nearest;
    },
    [holds, imageDimensions]
  );

  // Clear all holds
  const clearHolds = useCallback(() => {
    setHolds([]);
    setNextId(0);
  }, []);

  return {
    holds,
    addHold,
    removeHold,
    removeHoldById,
    removeLastHold,
    findHoldAt,
    clearHolds,
    loadHolds,
  };
}
