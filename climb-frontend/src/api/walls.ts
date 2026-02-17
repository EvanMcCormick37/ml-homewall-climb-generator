import { apiClient } from "./client";
import type { WallListResponse, WallDetail } from "@/types";

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
 * Get wall photo URL
 */
export function getWallPhotoUrl(wallId: string): string {
  const baseUrl =
    import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";
  return `${baseUrl}/walls/${wallId}/photo`;
}
