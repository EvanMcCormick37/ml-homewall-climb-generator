import { useState, useEffect, useCallback, useRef } from "react";
import { getLayouts, getLayout } from "@/api";
import type { LayoutMetadata, LayoutDetail } from "@/types";
import { fetchWithWakeRetry } from "./useWalls";

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

export function useLayout(
  layoutId: string,
  sizeId?: string
): UseLayoutReturn {
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
        { onWaking: () => setWaking(true), signal: controller.signal }
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
