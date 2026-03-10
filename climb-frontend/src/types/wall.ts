/**
 * Types for wall-related data structures
 *
 * These types mirror the Pydantic schemas in climb-api/app/schemas/
 * Keep in sync manually, or use Option 2 (openapi-typescript) for auto-generation.
 */

import type { HoldDetail } from "./holds";

export interface WallMetadata {
  id: string;
  name: string;
  photo_url: string;
  num_holds: number;
  num_climbs: number;
  dimensions: [number, number];
  angle: number | null;
  created_at: string;
  updated_at: string;
  owner_id: string;
  visibility: string;
  share_token?: string;
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

export type WallVisibility = "public" | "private" | "unlisted";

export interface WallCreate {
  name: string;
  photo: File;
  dimensions: [number, number];
  angle?: number;
  visibility?: WallVisibility;
}

export interface WallSetHolds {
  id: string;
  holds: HoldDetail[];
}
