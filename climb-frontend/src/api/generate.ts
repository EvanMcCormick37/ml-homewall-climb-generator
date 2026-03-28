import { apiClient } from "./client";
import type {
  GenerateRequest,
  GenerateSettings,
  GenerateResponse,
} from "@/types";

export async function generateClimbs(
  layoutId: string,
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
  if (request.x_offset != null) {
    params.x_offset = request.x_offset.toString();
  }
  if (settings) {
    params.timesteps = settings.timesteps.toString();
    params.guidance_value = settings.guidance_value.toString();
    params.t_start_projection = settings.t_start_projection.toString();
    params.deterministic = settings.deterministic.toString();
    params.seed = settings.seed.toString();
  }

  const response = await apiClient.get<GenerateResponse>(
    `/layouts/${layoutId}/generate`,
    { params },
  );
  return response.data;
}
