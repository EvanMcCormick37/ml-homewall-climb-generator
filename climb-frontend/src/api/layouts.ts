import { apiClient } from "./client";
import type {
  LayoutListResponse,
  LayoutDetail,
  LayoutCreate,
  LayoutCreateResponse,
  LayoutUpdate,
} from "@/types";
import type { HoldDetail } from "@/types";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

/**
 * Fetch all layouts (public + user's own if authenticated).
 */
export async function getLayouts(): Promise<LayoutListResponse> {
  const response = await apiClient.get<LayoutListResponse>("/layouts");
  return response.data;
}

/**
 * Fetch a single layout by ID, optionally filtered to a size's edge bounds.
 */
export async function getLayout(
  layoutId: string,
  sizeId?: string,
): Promise<LayoutDetail> {
  const params = sizeId ? { size_id: sizeId } : undefined;
  const response = await apiClient.get<LayoutDetail>(`/layouts/${layoutId}`, {
    params,
  });
  return response.data;
}

/**
 * Create a new layout (no photo — photo is uploaded separately via uploadLayoutPhoto).
 */
export async function createLayout(
  data: LayoutCreate,
): Promise<LayoutCreateResponse> {
  const formData = new FormData();
  formData.append("name", data.name);
  formData.append("dimensions", JSON.stringify(data.dimensions));
  if (data.default_angle != null)
    formData.append("default_angle", String(data.default_angle));
  if (data.description) formData.append("description", data.description);
  if (data.visibility) formData.append("visibility", data.visibility);

  const response = await apiClient.post<LayoutCreateResponse>(
    "/layouts",
    formData,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return response.data;
}

/**
 * Update editable fields on an existing layout.
 */
export async function updateLayout(
  layoutId: string,
  data: LayoutUpdate,
): Promise<void> {
  const formData = new FormData();
  if (data.name !== undefined) formData.append("name", data.name);
  if (data.description !== undefined)
    formData.append("description", data.description);
  if (data.visibility !== undefined)
    formData.append("visibility", data.visibility);
  if (data.default_angle !== undefined)
    formData.append(
      "default_angle",
      data.default_angle === null ? "" : String(data.default_angle),
    );
  if (data.image_edges !== undefined)
    formData.append("image_edges", JSON.stringify(data.image_edges));

  await apiClient.put(`/layouts/${layoutId}/edit`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}

/**
 * Upload or replace the photo for a layout.
 */
export async function uploadLayoutPhoto(
  layoutId: string,
  photo: File,
): Promise<void> {
  const formData = new FormData();
  formData.append("photo", photo);

  await apiClient.put(`/layouts/${layoutId}/photo`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}

/**
 * Delete a layout and all its sizes, holds, and climbs.
 */
export async function deleteLayout(layoutId: string): Promise<void> {
  await apiClient.delete(`/layouts/${layoutId}`);
}

/**
 * Set or replace the full hold set for a layout.
 */
export async function setLayoutHolds(
  layoutId: string,
  holds: HoldDetail[],
): Promise<{ id: string }> {
  const formData = new FormData();
  formData.append("holds", JSON.stringify(holds));

  const response = await apiClient.put<{ id: string }>(
    `/layouts/${layoutId}/holds`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return response.data;
}

/**
 * Get the photo URL for a layout.
 */
export function getLayoutPhotoUrl(layoutId: string): string {
  return `${BASE_URL}/layouts/${layoutId}/photo`;
}
