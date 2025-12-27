import { useState, useEffect, useCallback } from "react";
import { getClimbs, deleteClimb } from "@/api/climbs";
import type { Climb, ClimbFilters } from "@/types";

interface UseClimbsReturn {
  climbs: Climb[];
  total: number;
  loading: boolean;
  error: string | null;
  filters: ClimbFilters;
  setFilters: (filters: ClimbFilters) => void;
  refetch: () => Promise<void>;
  removeClimb: (climbId: string) => Promise<void>;
  selectedClimb: Climb | null;
  setSelectedClimb: (climb: Climb | null) => void;
}

/**
 * Custom hook for managing climbs for a wall.
 *
 * @param wallId - The wall ID to fetch climbs for
 * @param initialFilters - Optional initial filter settings
 * @returns Climb state, filtering, and CRUD operations
 */
export function useClimbs(
  wallId: string,
  initialFilters: ClimbFilters = {}
): UseClimbsReturn {
  const [climbs, setClimbs] = useState<Climb[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<ClimbFilters>(initialFilters);
  const [selectedClimb, setSelectedClimb] = useState<Climb | null>(null);

  const fetchClimbs = useCallback(async () => {
    if (!wallId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await getClimbs(wallId, filters);
      setClimbs(response.climbs);
      setTotal(response.total);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch climbs";
      setError(message);
      console.error("Error fetching climbs:", err);
    } finally {
      setLoading(false);
    }
  }, [wallId, filters]);

  useEffect(() => {
    fetchClimbs();
  }, [fetchClimbs]);

  const removeClimb = useCallback(
    async (climbId: string): Promise<void> => {
      await deleteClimb(wallId, climbId);
      // Update local state
      setClimbs((prev) => prev.filter((c) => c.id !== climbId));
      setTotal((prev) => prev - 1);
      // Clear selection if deleted climb was selected
      if (selectedClimb?.id === climbId) {
        setSelectedClimb(null);
      }
    },
    [wallId, selectedClimb]
  );

  return {
    climbs,
    total,
    loading,
    error,
    filters,
    setFilters,
    refetch: fetchClimbs,
    removeClimb,
    selectedClimb,
    setSelectedClimb,
  };
}
