import { useState, useEffect, useCallback } from "react";
import { getWalls, createWall, deleteWall } from "@/api";
import type { WallMetadata, WallCreate, WallCreateResponse } from "@/types";

interface UseWallsReturn {
  walls: WallMetadata[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  createNewWall: (data: WallCreate) => Promise<WallCreateResponse>;
  removeWall: (wallId: string) => Promise<void>;
}

export function useWalls(): UseWallsReturn {
  const [walls, setWalls] = useState<WallMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWalls = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getWalls();
      setWalls(response.walls);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch walls";
      setError(message);
      console.error("Error fetching walls:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWalls();
  }, [fetchWalls]);

  const createNewWall = useCallback(
    async (data: WallCreate): Promise<WallCreateResponse> => {
      const response = await createWall(data);
      // Refetch walls after creation
      await fetchWalls();
      return response;
    },
    [fetchWalls]
  );

  const removeWall = useCallback(async (wallId: string): Promise<void> => {
    await deleteWall(wallId);
    // Update local state
    setWalls((prev) => prev.filter((w) => w.id !== wallId));
  }, []);

  return {
    walls,
    loading,
    error,
    refetch: fetchWalls,
    createNewWall,
    removeWall,
  };
}
