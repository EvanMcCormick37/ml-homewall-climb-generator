/**
 * Types for climb generation via the DDPM endpoint.
 */

export type GradeScale = "v_grade" | "font";

export interface GenerateRequest {
  num_climbs: number;
  grade: string;
  grade_scale: GradeScale;
  angle?: number | null;
  deterministic: boolean;
}

export interface GeneratedClimb {
  holds: number[];
  num_holds: number;
}

export interface GenerateResponse {
  wall_id: string;
  climbs: GeneratedClimb[];
  num_generated: number;
  parameters: GenerateRequest;
}
