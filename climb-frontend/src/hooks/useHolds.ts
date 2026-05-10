import { useState, useCallback, useMemo } from "react";
import type { HoldDetail, Tag } from "@/types";
import { buildWallHomography, holdToPixel, pixelToFeet, type Dimensions } from "@/utils/coordinateSpace";

export function useHolds(
  imageDimensions: Dimensions,
  wallDimensions: Dimensions,
  imageEdges?: [number, number, number, number] | null,
  homographySrcCorners?: number[] | null,
) {
  const [holds, setHolds] = useState<HoldDetail[]>([]);
  const maxHoldIndex = useMemo(
    () => (holds.length > 0 ? Math.max(...holds.map((h) => h.hold_index)) : 0),
    [holds]
  );

  const { H, Hinv } = useMemo(
    () => buildWallHomography(homographySrcCorners, imageDimensions, wallDimensions),
    [homographySrcCorners, imageDimensions, wallDimensions]
  );

  const toFeetCoords = useCallback(
    (pixelX: number, pixelY: number) =>
      pixelToFeet(pixelX, pixelY, H, imageEdges, imageDimensions, wallDimensions),
    [H, imageEdges, imageDimensions, wallDimensions]
  );

  const toPixelCoords = useCallback(
    (hold: Pick<HoldDetail, "x" | "y">) =>
      holdToPixel(hold, Hinv, imageEdges, imageDimensions, wallDimensions),
    [Hinv, imageEdges, imageDimensions, wallDimensions]
  );

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
        setHolds((prev) => prev.filter((_, i) => i !== closestIndex));
      }
    },
    [holds, toPixelCoords]
  );

  const removeHoldByIndex = useCallback((holdIndex: number) => {
    setHolds((prev) => prev.filter((h) => h.hold_index !== holdIndex));
  }, []);

  const removeLastHold = useCallback(() => {
    setHolds((prev) => (prev.length === 0 ? prev : prev.slice(0, -1)));
  }, []);

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

  const clearHolds = useCallback(() => setHolds([]), []);

  const loadHolds = useCallback((holdsData: HoldDetail[]) => setHolds(holdsData), []);

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
