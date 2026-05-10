import { useState, useEffect, useCallback, useRef } from "react";
import { getLayouts, getLayout, is502 } from "@/api";
import type { LayoutMetadata, LayoutDetail } from "@/types";

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
    signal?.throwIfAborted();
    try {
      return await fetcher();
    } catch (err) {
      if (is502(err) && Date.now() < deadline) {
        onWaking?.();
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, retryIntervalMs);
          signal?.addEventListener(
            "abort",
            () => {
              clearTimeout(timer);
              reject(signal.reason);
            },
            { once: true },
          );
        });
      } else {
        throw err;
      }
    }
  }
}

/**
 * Generic hook that fetches data with 502-wake retry, abort on unmount,
 * and loading/waking/error state. Callers pass a fetcher and its deps.
 */
function useWakeRetry<T>(
  fetcher: () => Promise<T>,
  // eslint-disable-next-line react-hooks/exhaustive-deps
  deps: React.DependencyList,
): { data: T | null; loading: boolean; waking: boolean; error: string | null; refetch: () => Promise<void> } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Keep a ref to the latest fetcher so the stable callback always calls it fresh
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const refetch = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setWaking(false);
    setError(null);
    try {
      const result = await fetchWithWakeRetry(() => fetcherRef.current(), {
        onWaking: () => setWaking(true),
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setData(result);
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof Error ? err.message : "Fetch failed");
      console.error("Fetch error:", err);
    } finally {
      if (!controller.signal.aborted) {
        setWaking(false);
        setLoading(false);
      }
    }
  }, deps); // deps controls when a re-fetch is triggered

  useEffect(() => {
    refetch();
    return () => abortRef.current?.abort();
  }, [refetch]);

  return { data, loading, waking, error, refetch };
}

export function useLayouts() {
  const { data, loading, waking, error, refetch } = useWakeRetry(
    () => getLayouts().then((r) => r.layouts),
    [],
  );
  return { layouts: data ?? [], loading, waking, error, refetch };
}

interface UseLayoutReturn {
  layout: LayoutDetail | null;
  loading: boolean;
  waking: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useLayout(layoutId: string, sizeId?: string): UseLayoutReturn {
  const { data, loading, waking, error, refetch } = useWakeRetry(
    () => getLayout(layoutId, sizeId),
    [layoutId, sizeId],
  );
  return { layout: data ?? null, loading, waking, error, refetch };
}
