import { useState, useCallback, useMemo } from "react";
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
  const maxHoldIndex = useMemo(
    () => (holds.length > 0 ? Math.max(...holds.map((h) => h.hold_index)) : 0),
    [holds]
  );
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
      holdIndex?: number,
      pull_x?: number,
      pull_y?: number,
      useability?: number,
      is_foot?: number
    ) => {
      const { x, y } = toFeetCoords(pixelX, pixelY);
      const newHold: HoldDetail = {
        x,
        y,
        hold_index: holdIndex ?? maxHoldIndex + 1,
        pull_x: pull_x ?? null,
        pull_y: pull_y ?? null,
        useability: useability ?? null,
        is_foot: is_foot ?? 0,
      };

      setHolds((prev) => [...prev, newHold]);
    },
    [holds.length, toFeetCoords]
  );

  const updateHold = useCallback(
    (holdIndex: number, updates: Partial<HoldDetail>) => {
      setHolds((prev) =>
        prev.map((hold) =>
          hold.hold_index === holdIndex ? { ...hold, ...updates } : hold
        )
      );
    },
    []
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
          // DO NOT Re-index remaining holds
          return updated;
        });
      }
    },
    [holds, toPixelCoords]
  );

  // Remove hold by index
  const removeHoldByIndex = useCallback((holdIndex: number) => {
    setHolds((prev) => {
      const updated = prev.filter((h) => h.hold_index !== holdIndex);
      // DO NOT re-index remaining holds
      return updated;
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
    updateHold,
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
