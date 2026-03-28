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
  x_offset: number | null;
}

export interface GenerateSettings {
  timesteps: number;
  guidance_value: number;
  t_start_projection: number;
  deterministic: boolean;
  seed: number;
}

export const DEFAULT_GENERATE_SETTINGS: GenerateSettings = {
  timesteps: 100,
  guidance_value: 5.0,
  t_start_projection: 0.5,
  deterministic: false,
  seed: 37,
};

export const FAST_GENERATE_SETTINGS: GenerateSettings = {
  timesteps: 50,
  guidance_value: 5.0,
  t_start_projection: 0.5,
  deterministic: false,
  seed: 37,
};

export const SLOW_GENERATE_SETTINGS: GenerateSettings = {
  timesteps: 200,
  guidance_value: 5.0,
  t_start_projection: 0.5,
  deterministic: false,
  seed: 37,
};

export interface GenerateResponse {
  wall_id: string;
  climbs: Holdset[];
  num_generated: number;
  parameters: GenerateRequest;
}
