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

export function useLayouts() {
  const [layouts, setLayouts] = useState<LayoutMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchLayouts = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setWaking(false);
    setError(null);

    try {
      const response = await fetchWithWakeRetry(() => getLayouts(), {
        onWaking: () => setWaking(true),
        signal: controller.signal,
      });
      setLayouts(response.layouts);
    } catch (err) {
      if (controller.signal.aborted) return;
      const message =
        err instanceof Error ? err.message : "Failed to fetch layouts";
      setError(message);
      console.error("Error fetching layouts:", err);
    } finally {
      if (!controller.signal.aborted) {
        setWaking(false);
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchLayouts();
    return () => abortRef.current?.abort();
  }, [fetchLayouts]);

  return { layouts, loading, waking, error, refetch: fetchLayouts };
}

interface UseLayoutReturn {
  layout: LayoutDetail | null;
  loading: boolean;
  waking: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useLayout(layoutId: string, sizeId?: string): UseLayoutReturn {
  const [layout, setLayout] = useState<LayoutDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchLayout = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setWaking(false);
    setError(null);

    try {
      const response = await fetchWithWakeRetry(
        () => getLayout(layoutId, sizeId),
        { onWaking: () => setWaking(true), signal: controller.signal },
      );
      setLayout(response);
    } catch (err) {
      if (controller.signal.aborted) return;
      const message =
        err instanceof Error ? err.message : "Failed to fetch layout";
      setError(message);
      console.error("Error fetching layout:", err);
    } finally {
      if (!controller.signal.aborted) {
        setWaking(false);
        setLoading(false);
      }
    }
  }, [layoutId, sizeId]);

  useEffect(() => {
    fetchLayout();
    return () => abortRef.current?.abort();
  }, [fetchLayout]);

  return { layout, loading, waking, error, refetch: fetchLayout };
}
