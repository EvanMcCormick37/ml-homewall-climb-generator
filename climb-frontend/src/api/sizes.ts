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
 * Create a new size for a layout.
 */
export async function createSize(
  layoutId: string,
  data: SizeCreate
): Promise<SizeCreateResponse> {
  const formData = new FormData();
  formData.append("name", data.name);
  formData.append("edges", JSON.stringify(data.edges));
  formData.append("kickboard", String(data.kickboard));

  const response = await apiClient.post<SizeCreateResponse>(
    `/layouts/${layoutId}/sizes`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return response.data;
}

/**
 * Delete a size.
 */
export async function deleteSize(
  layoutId: string,
  sizeId: string
): Promise<void> {
  await apiClient.delete(`/layouts/${layoutId}/sizes/${sizeId}`);
}
