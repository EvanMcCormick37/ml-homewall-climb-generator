/**
 * Types for climb generation via the DDPM.
 *
 * Mirrors the Pydantic schemas in climb-api/app/schemas/generate.py
 */
import type { Holdset } from "./climb";

export type GradeScale = "v_grade" | "font";

export interface GenerateRequest {
  num_climbs: number;
  grade: string;
  grade_scale: GradeScale;
  angle: number | null;
  deterministic: boolean;
}

export interface GenerateResponse {
  wall_id: string;
  climbs: Holdset[];
  num_generated: number;
  parameters: GenerateRequest;
}
