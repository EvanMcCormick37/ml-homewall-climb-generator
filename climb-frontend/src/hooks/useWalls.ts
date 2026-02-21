import { useState, useEffect, useCallback } from "react";
import { getWalls } from "@/api";
import type { WallMetadata } from "@/types";

const RETRY_INTERVAL_MS = 3000;
const MAX_RETRY_DURATION_MS = 20000;

interface UseWallsReturn {
  walls: WallMetadata[];
  loading: boolean;
  waking: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

function is502(err: unknown): boolean {
  if (err && typeof err === "object") {
    // Axios error shape
    const axiosErr = err as {
      response?: { status?: number };
      message?: string;
    };
    if (axiosErr.response?.status === 502) return true;
    // Fallback: check message string
    if (axiosErr.message && /502/.test(axiosErr.message)) return true;
  }
  return false;
}

export function useWalls(): UseWallsReturn {
  const [walls, setWalls] = useState<WallMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWalls = useCallback(async () => {
    setLoading(true);
    setWaking(false);
    setError(null);

    const deadline = Date.now() + MAX_RETRY_DURATION_MS;

    while (true) {
      try {
        const response = await getWalls();
        setWalls(response.walls);
        setWaking(false);
        setLoading(false);
        return;
      } catch (err) {
        if (is502(err) && Date.now() < deadline) {
          // Server is asleep — flip the waking flag and retry after a pause
          setWaking(true);
          await new Promise<void>((resolve) =>
            setTimeout(resolve, RETRY_INTERVAL_MS),
          );
          // continue the while loop
        } else {
          // Either not a 502, or we've exceeded the deadline
          const message =
            err instanceof Error ? err.message : "Failed to fetch walls";
          setError(message);
          console.error("Error fetching walls:", err);
          setWaking(false);
          setLoading(false);
          return;
        }
      }
    }
  }, []);

  useEffect(() => {
    fetchWalls();
  }, [fetchWalls]);

  return {
    walls,
    loading,
    waking,
    error,
    refetch: fetchWalls,
  };
}
