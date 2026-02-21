import { useState, useEffect, useCallback, useRef } from "react";
import { getWalls, getWall } from "@/api";
import type { WallMetadata, WallDetail } from "@/types";

interface UseWallsReturn {
  walls: WallMetadata[];
  loading: boolean;
  waking: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

interface UseWallsReturn {
  walls: WallMetadata[];
  loading: boolean;
  waking: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}
import { is502 } from "@/api";

const DEFAULT_RETRY_INTERVAL_MS = 3000;
const DEFAULT_MAX_RETRY_DURATION_MS = 20000;

export interface WakeRetryOptions {
  retryIntervalMs?: number;
  maxRetryDurationMs?: number;
  /** Called when the first 502 is detected (server is waking). */
  onWaking?: () => void;
  /** AbortSignal so callers can cancel retries (e.g. on unmount). */
  signal?: AbortSignal;
}

/**
 * Execute an async fetcher, automatically retrying on 502 (server waking)
 * until the request succeeds or the retry window expires.
 */
export async function fetchWithWakeRetry<T>(
  fetcher: () => Promise<T>,
  options: WakeRetryOptions = {},
): Promise<T> {
  const {
    retryIntervalMs = DEFAULT_RETRY_INTERVAL_MS,
    maxRetryDurationMs = DEFAULT_MAX_RETRY_DURATION_MS,
    onWaking,
    signal,
  } = options;

  const deadline = Date.now() + maxRetryDurationMs;

  while (true) {
    // Bail out immediately if the caller has cancelled
    signal?.throwIfAborted();

    try {
      return await fetcher();
    } catch (err) {
      if (is502(err) && Date.now() < deadline) {
        onWaking?.();
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, retryIntervalMs);
          // If the signal fires while we're sleeping, clean up and reject
          signal?.addEventListener(
            "abort",
            () => {
              clearTimeout(timer);
              reject(signal.reason);
            },
            { once: true },
          );
        });
        // loop back to retry
      } else {
        throw err;
      }
    }
  }
}

export function useWalls(): UseWallsReturn {
  const [walls, setWalls] = useState<WallMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchWalls = useCallback(async () => {
    // Abort any in-flight request before starting a new one
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setWaking(false);
    setError(null);

    try {
      const response = await fetchWithWakeRetry(() => getWalls(), {
        onWaking: () => setWaking(true),
        signal: controller.signal,
      });
      setWalls(response.walls);
    } catch (err) {
      // Silently ignore aborted requests (component unmounted or refetch called)
      if (controller.signal.aborted) return;
      const message =
        err instanceof Error ? err.message : "Failed to fetch walls";
      setError(message);
      console.error("Error fetching walls:", err);
    } finally {
      if (!controller.signal.aborted) {
        setWaking(false);
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchWalls();
    return () => abortRef.current?.abort();
  }, [fetchWalls]);

  return { walls, loading, waking, error, refetch: fetchWalls };
}

interface UseWallReturn {
  wall: WallDetail | null;
  loading: boolean;
  waking: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useWall(wallId: string): UseWallReturn {
  const [wall, setWall] = useState<WallDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchWall = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setWaking(false);
    setError(null);

    try {
      const response = await fetchWithWakeRetry(() => getWall(wallId), {
        onWaking: () => setWaking(true),
        signal: controller.signal,
      });
      setWall(response);
    } catch (err) {
      if (controller.signal.aborted) return;
      const message =
        err instanceof Error ? err.message : "Failed to fetch wall";
      setError(message);
      console.error("Error fetching wall:", err);
    } finally {
      if (!controller.signal.aborted) {
        setWaking(false);
        setLoading(false);
      }
    }
  }, [wallId]);

  useEffect(() => {
    fetchWall();
    return () => abortRef.current?.abort();
  }, [fetchWall]);

  return { wall, loading, waking, error, refetch: fetchWall };
}
