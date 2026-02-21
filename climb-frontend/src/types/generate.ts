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
}

export interface GenerateSettings {
  timesteps: number;
  t_start_projection: number;
  x_offset: number | null;
  deterministic: boolean;
  seed: number;
}

export const DEFAULT_GENERATE_SETTINGS: GenerateSettings = {
  timesteps: 25,
  t_start_projection: 0.0,
  x_offset: null,
  deterministic: false,
  seed: 37,
};

export const FAST_GENERATE_SETTINGS: GenerateSettings = {
  timesteps: 10,
  t_start_projection: 0.0,
  x_offset: null,
  deterministic: false,
  seed: 37,
};

export const SLOW_GENERATE_SETTINGS: GenerateSettings = {
  timesteps: 100,
  t_start_projection: 1.0,
  x_offset: null,
  deterministic: false,
  seed: 37,
};

export interface GenerateResponse {
  wall_id: string;
  climbs: Holdset[];
  num_generated: number;
  parameters: GenerateRequest;
}
