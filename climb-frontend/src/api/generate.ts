import { apiClient } from "./client";
import type { GenerateRequest, GenerateResponse } from "@/types";

export async function generateClimbs(
  wallId: string,
  request: GenerateRequest,
): Promise<GenerateResponse> {
  const params: Record<string, string> = {
    num_climbs: request.num_climbs.toString(),
    grade: request.grade,
    grade_scale: request.grade_scale,
    deterministic: request.deterministic.toString(),
  };
  if (request.angle != null) {
    params.angle = request.angle.toString();
  }

  const response = await apiClient.get<GenerateResponse>(
    `/walls/${wallId}/generate`,
    { params },
  );
  return response.data;
}
