import { useState, useEffect, useCallback } from "react";
import { getWalls } from "@/api";
import type { WallMetadata } from "@/types";

interface UseWallsReturn {
  walls: WallMetadata[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
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

  return {
    walls,
    loading,
    error,
    refetch: fetchWalls,
  };
}
