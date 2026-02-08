import { apiClient } from "./client";
import type { GenerateRequest, GenerateResponse } from "@/types";

/**
 * Generate climbs for a wall using the DDPM model.
 */
export async function generateClimbs(
  wallId: string,
  request: GenerateRequest
): Promise<GenerateResponse> {
  const response = await apiClient.post<GenerateResponse>(
    `/walls/${wallId}/generate`,
    request
  );
  return response.data;
}
