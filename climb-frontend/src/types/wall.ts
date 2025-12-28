/**
 * Types for wall-related data structures
 *
 * These types mirror the Pydantic schemas in climb-api/app/schemas/
 * Keep in sync manually, or use Option 2 (openapi-typescript) for auto-generation.
 */

// From schemas/base.py
export interface HoldDetail {
  hold_index: number;
  x: number; // horizontal position
  y: number; // vertival position
  pull_x: number | null; // -1 to 1, pull direction
  pull_y: number | null; // -1 to 1, pull direction
  useability: number | null; // 0-1 or null
}

export type HoldMode = "add" | "remove" | "select";

export interface WallMetadata {
  id: string;
  name: string;
  photo_url: string;
  num_holds: number;
  num_climbs: number;
  num_models: number;
  dimensions: [number, number];
  angle: number | null;
  created_at: string;
  updated_at: string;
}

export interface WallDetail {
  metadata: WallMetadata;
  holds: HoldDetail[];
}

export interface WallListResponse {
  walls: WallMetadata[];
  total: number;
}

export interface WallCreateResponse {
  id: string;
  name: string;
}

export interface WallCreate {
  name: string;
  photo: File;
  dimensions: [number, number];
  angle: number | null;
}

export interface WallSetHolds {
  id: string;
  holds: HoldDetail[];
}
