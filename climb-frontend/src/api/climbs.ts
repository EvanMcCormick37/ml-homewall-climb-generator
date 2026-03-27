import { apiClient } from "./client";
import {
  type ClimbListResponse,
  type ClimbCreate,
  type ClimbCreateResponse,
  type ClimbDeleteResponse,
  type ClimbFilters,
  DEFAULT_CLIMB_FILTERS,
} from "@/types";

/**
 * Fetch climbs for a layout with optional filters
 */
export async function getClimbs(
  layoutId: string,
  filters: ClimbFilters = DEFAULT_CLIMB_FILTERS,
): Promise<ClimbListResponse> {
  const params = new URLSearchParams();

  params.append("grade_scale", filters.gradeScale);
  params.append("min_grade", filters.minGrade);
  params.append("max_grade", filters.maxGrade);
  if (filters.includeProjects !== undefined) {
    params.append("include_projects", String(filters.includeProjects));
  }
  if (filters.setterName) {
    params.append("setter_name", filters.setterName);
  }
  if (filters.nameIncludes) {
    params.append("name_includes", filters.nameIncludes);
  }
  if (filters.holdsInclude) {
    filters.holdsInclude.forEach((h) =>
      params.append("holds_include", h.toString()),
    );
  }
  if (filters.tagsInclude) {
    filters.tagsInclude.forEach((t) => params.append("tags_include", t));
  }
  if (filters.after) {
    params.append("after", filters.after);
  }
  if (filters.sortBy) {
    params.append("sort_by", filters.sortBy);
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
  const url = `/layouts/${layoutId}/climbs${queryString ? `?${queryString}` : ""}`;

  const response = await apiClient.get<ClimbListResponse>(url);
  return response.data;
}

/**
 * Create a new climb
 */
export async function createClimb(
  layoutId: string,
  data: ClimbCreate,
): Promise<ClimbCreateResponse> {
  // Validate holds before sending
  if (data.holdset.start.length === 0) {
    throw new Error("At least one start hold is required");
  }
  if (data.holdset.start.length > 2) {
    throw new Error("Maximum of 2 start holds allowed");
  }
  if (data.holdset.finish.length === 0) {
    throw new Error("At least one finish hold is required");
  }
  if (data.holdset.finish.length > 2) {
    throw new Error("Maximum of 2 finish holds allowed");
  }

  const response = await apiClient.post<ClimbCreateResponse>(
    `/layouts/${layoutId}/climbs`,
    data,
  );

  return response.data;
}

/**
 * Delete a climb
 */
export async function deleteClimb(
  layoutId: string,
  climbId: string,
): Promise<ClimbDeleteResponse> {
  const response = await apiClient.delete<ClimbDeleteResponse>(
    `/layouts/${layoutId}/climbs/${climbId}`,
  );
  return response.data;
}
