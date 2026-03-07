import { apiClient } from "./client";
import type { SizeMetadata, SizeCreate, SizeCreateResponse } from "@/types";

/**
 * Fetch all sizes for a layout.
 */
export async function getSizes(layoutId: string): Promise<SizeMetadata[]> {
  const response = await apiClient.get<SizeMetadata[]>(
    `/layouts/${layoutId}/sizes`
  );
  return response.data;
}

/**
 * Create a new size for a layout, optionally with a photo.
 */
export async function createSize(
  layoutId: string,
  data: SizeCreate
): Promise<SizeCreateResponse> {
  const formData = new FormData();
  formData.append("name", data.name);
  if (data.width_ft !== undefined)
    formData.append("width_ft", String(data.width_ft));
  if (data.height_ft !== undefined)
    formData.append("height_ft", String(data.height_ft));
  if (data.edge_left !== undefined)
    formData.append("edge_left", String(data.edge_left));
  if (data.edge_right !== undefined)
    formData.append("edge_right", String(data.edge_right));
  if (data.edge_bottom !== undefined)
    formData.append("edge_bottom", String(data.edge_bottom));
  if (data.edge_top !== undefined)
    formData.append("edge_top", String(data.edge_top));
  if (data.photo) formData.append("photo", data.photo);

  const response = await apiClient.post<SizeCreateResponse>(
    `/layouts/${layoutId}/sizes`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return response.data;
}

/**
 * Upload or replace the photo for an existing size.
 */
export async function uploadSizePhoto(
  layoutId: string,
  sizeId: string,
  photo: File
): Promise<void> {
  const formData = new FormData();
  formData.append("photo", photo);

  await apiClient.put(
    `/layouts/${layoutId}/sizes/${sizeId}/photo`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
}

/**
 * Delete a size and its photo.
 */
export async function deleteSize(
  layoutId: string,
  sizeId: string
): Promise<void> {
  await apiClient.delete(`/layouts/${layoutId}/sizes/${sizeId}`);
}
