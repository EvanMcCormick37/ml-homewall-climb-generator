import { useState, useCallback, useMemo } from "react";
import type { HoldDetail, Tag } from "@/types";

interface Dimensions {
  width: number;
  height: number;
}

export function useHolds(
  imageDimensions: Dimensions,
  wallDimensions: Dimensions,
  imageEdges?: [number, number, number, number] | null
) {
  const [holds, setHolds] = useState<HoldDetail[]>([]);
  const maxHoldIndex = useMemo(
    () => (holds.length > 0 ? Math.max(...holds.map((h) => h.hold_index)) : 0),
    [holds]
  );

  // Resolve image_edges, falling back to full wall dimensions
  const [imgL, imgR, imgB, imgT] = imageEdges ?? [
    0,
    wallDimensions.width,
    0,
    wallDimensions.height,
  ];

  // Convert pixel coordinates to feet
  const toFeetCoords = useCallback(
    (pixelX: number, pixelY: number) => {
      const xFeet = imgL + (pixelX / imageDimensions.width) * (imgR - imgL);
      const yFeet =
        imgB +
        ((imageDimensions.height - pixelY) / imageDimensions.height) *
          (imgT - imgB);
      return { x: xFeet, y: yFeet };
    },
    [imageDimensions, imgL, imgR, imgB, imgT]
  );

  // Convert feet coordinates to pixels
  const toPixelCoords = useCallback(
    (hold: HoldDetail) => {
      const pixelX = ((hold.x - imgL) / (imgR - imgL)) * imageDimensions.width;
      const pixelY =
        ((imgT - hold.y) / (imgT - imgB)) * imageDimensions.height;
      return { x: pixelX, y: pixelY };
    },
    [imageDimensions, imgL, imgR, imgB, imgT]
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
      is_foot?: boolean,
      tags?: Tag[]
    ): number => {
      const { x, y } = toFeetCoords(pixelX, pixelY);
      const newHold: HoldDetail = {
        x,
        y,
        hold_index: holdIndex ?? maxHoldIndex + 1,
        pull_x: pull_x ?? null,
        pull_y: pull_y ?? null,
        useability: useability ?? null,
        is_foot: is_foot ?? false,
        tags: tags ?? [],
      };

      setHolds((prev) => [...prev, newHold]);
      return newHold.hold_index;
    },
    [holds.length, toFeetCoords, maxHoldIndex]
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
