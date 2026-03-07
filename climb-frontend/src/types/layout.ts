/**
 * Types for layout/size data structures.
 *
 * A layout is a unique hold arrangement (replaces the old Wall concept).
 * A size is a physical dimension variant of a layout (different edges/kickboard).
 * Climbs and holds are tied to a layout, not a size.
 * Photos are owned by the layout, not the size.
 */
import type { HoldDetail } from "./wall";

export interface SizeMetadata {
  id: string;
  layout_id: string;
  name: string;
  edges: number[]; // [left, right, bottom, top] in feet
  kickboard: boolean;
  created_at: string;
  updated_at: string;
}

export interface LayoutMetadata {
  id: string;
  name: string;
  description: string | null;
  dimensions: number[]; // [width_ft, height_ft]
  default_angle: number | null;
  sizes: SizeMetadata[];
  owner_id: string;
  visibility: "public" | "private" | "unlisted";
  share_token: string | null;
  created_at: string;
  updated_at: string;
}

export interface LayoutDetail {
  metadata: LayoutMetadata;
  holds: HoldDetail[];
}

export interface LayoutListResponse {
  layouts: LayoutMetadata[];
  total: number;
}

export interface LayoutCreate {
  name: string;
  dimensions: number[]; // [width_ft, height_ft]
  default_angle?: number | null;
  description?: string;
  visibility?: "public" | "private" | "unlisted";
}

export interface LayoutCreateResponse {
  id: string;
  name: string;
}

export interface SizeCreate {
  name: string;
  edges: number[]; // [left, right, bottom, top] in feet
  kickboard: boolean;
}

export interface SizeCreateResponse {
  id: string;
  layout_id: string;
  name: string;
}
