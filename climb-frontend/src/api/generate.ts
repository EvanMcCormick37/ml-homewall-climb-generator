import { apiClient } from "./client";
import type {
  GenerateRequest,
  GenerateSettings,
  GenerateResponse,
} from "@/types";

export async function generateClimbs(
  wallId: string,
  request: GenerateRequest,
  settings?: GenerateSettings,
): Promise<GenerateResponse> {
  const params: Record<string, string> = {
    num_climbs: request.num_climbs.toString(),
    grade: request.grade,
    grade_scale: request.grade_scale,
  };
  if (request.angle != null) {
    params.angle = request.angle.toString();
  }
  if (settings) {
    params.timesteps = settings.timesteps.toString();
    params.t_start_projection = settings.t_start_projection.toString();
    if (settings.x_offset != null) {
      params.x_offset = settings.x_offset.toString();
    }
    params.deterministic = settings.deterministic.toString();
  }

  const response = await apiClient.get<GenerateResponse>(
    `/walls/${wallId}/generate`,
    { params },
  );
  return response.data;
}
