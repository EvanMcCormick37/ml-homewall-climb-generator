/**
 * Types for wall-related data structures
 *
 * These types mirror the Pydantic schemas in climb-api/app/schemas/
 * Keep in sync manually, or use Option 2 (openapi-typescript) for auto-generation.
 */

export interface HoldDetail {
  hold_index: number;
  x: number;
  y: number;
  pull_x: number | null;
  pull_y: number | null;
  useability: number | null;
  is_foot: number;
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

export interface EnabledFeatures {
  direction: boolean;
  useability: boolean;
  footholds: boolean;
}

export type FeatureLabel = "direction" | "useability" | "footholds";
