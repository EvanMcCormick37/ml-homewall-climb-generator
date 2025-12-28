import { apiClient } from "./client";
import type {
  ClimbListResponse,
  ClimbCreate,
  ClimbCreateResponse,
  ClimbDeleteResponse,
  ClimbFilters,
} from "@/types";

/**
 * Fetch climbs for a wall with optional filters
 */
export async function getClimbs(
  wallId: string,
  filters: ClimbFilters = {}
): Promise<ClimbListResponse> {
  const params = new URLSearchParams();

  if (filters.grade_range) {
    params.append("grade_range", filters.grade_range[0].toString());
    params.append("grade_range", filters.grade_range[1].toString());
  }
  if (filters.include_projects !== undefined) {
    params.append("include_projects", String(filters.include_projects));
  }
  if (filters.setter) {
    params.append("setter", filters.setter);
  }
  if (filters.name_includes) {
    params.append("name_includes", filters.name_includes);
  }
  if (filters.holds_include) {
    filters.holds_include.forEach((h) =>
      params.append("holds_include", h.toString())
    );
  }
  if (filters.tags_include) {
    filters.tags_include.forEach((t) => params.append("tags_include", t));
  }
  if (filters.after) {
    params.append("after", filters.after);
  }
  if (filters.sort_by) {
    params.append("sort_by", filters.sort_by);
  }
  if (filters.descending !== undefined) {
    params.append("descending", String(filters.descending));
  }
  if (filters.limit !== undefined) {
    params.append("limit", filters.limit.toString());
  }
  if (filters.offset !== undefined) {
    params.append("offset", filters.offset.toString());
  }

  const queryString = params.toString();
  const url = `/walls/${wallId}/climbs${queryString ? `?${queryString}` : ""}`;

  const response = await apiClient.get<ClimbListResponse>(url);
  return response.data;
}

/**
 * Create a new climb
 */
export async function createClimb(
  wallId: string,
  data: ClimbCreate
): Promise<ClimbCreateResponse> {
  // Validate holds before sending
  if (data.holds.start.length === 0) {
    throw new Error("At least one start hold is required");
  }
  if (data.holds.start.length > 2) {
    throw new Error("Maximum of 2 start holds allowed");
  }
  if (data.holds.finish.length === 0) {
    throw new Error("At least one finish hold is required");
  }
  if (data.holds.finish.length > 2) {
    throw new Error("Maximum of 2 finish holds allowed");
  }

  const response = await apiClient.post<ClimbCreateResponse>(
    `/walls/${wallId}/climbs`,
    {
      name: data.name,
      grade: data.grade,
      setter: data.setter,
      tags: data.tags,
      // Send holds in the new format
      holds: data.holds,
    }
  );

  return response.data;
}

/**
 * Delete a climb
 */
export async function deleteClimb(
  wallId: string,
  climbId: string
): Promise<ClimbDeleteResponse> {
  const response = await apiClient.delete<ClimbDeleteResponse>(
    `/walls/${wallId}/climbs/${climbId}`
  );
  return response.data;
}
