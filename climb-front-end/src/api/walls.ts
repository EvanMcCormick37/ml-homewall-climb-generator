import { apiClient } from "./client";
import type {
  WallListResponse,
  WallDetail,
  WallCreateResponse,
  WallCreate,
} from "@/types";

/**
 * Fetch all walls
 */
export async function getWalls(): Promise<WallListResponse> {
  const response = await apiClient.get<WallListResponse>("/walls");
  return response.data;
}

/**
 * Fetch a single wall by ID
 */
export async function getWall(wallId: string): Promise<WallDetail> {
  const response = await apiClient.get<WallDetail>(`/walls/${wallId}`);
  return response.data;
}

/**
 * Create a new wall (metadata + photo, no holds)
 */
export async function createWall(
  data: WallCreate
): Promise<WallCreateResponse> {
  const formData = new FormData();
  formData.append("name", data.name);
  formData.append("photo", data.photo);
  if (data.dimensions) {
    formData.append(
      "dimensions",
      `${data.dimensions[0]},${data.dimensions[1]}`
    );
  }
  if (data.angle !== undefined) {
    formData.append("angle", String(data.angle));
  }

  const response = await apiClient.post<WallCreateResponse>(
    "/walls",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );
  return response.data;
}

/**
 * Delete a wall
 */
export async function deleteWall(wallId: string): Promise<void> {
  await apiClient.delete(`/walls/${wallId}`);
}

/**
 * Get wall photo URL
 */
export function getWallPhotoUrl(wallId: string): string {
  const baseUrl =
    import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";
  return `${baseUrl}/walls/${wallId}/photo`;
}
