/**
 * Types for layout/size data structures.
 *
 * A layout is a unique hold arrangement (replaces the old Wall concept).
 * A size is a physical dimension variant of a layout (different photo/dimensions).
 * Climbs and holds are tied to a layout, not a size.
 */
import type { HoldDetail } from "./wall";

export interface SizeMetadata {
  id: string;
  layout_id: string;
  name: string;
  width_ft: number | null;
  height_ft: number | null;
  edge_left: number;
  edge_right: number | null;
  edge_bottom: number;
  edge_top: number | null;
  photo_url: string | null;
  num_climbs: number;
  created_at: string;
  updated_at: string;
}

export interface LayoutMetadata {
  id: string;
  name: string;
  description: string | null;
  num_holds: number;
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
  description?: string;
  visibility?: "public" | "private" | "unlisted";
}

export interface LayoutCreateResponse {
  id: string;
  name: string;
}

export interface SizeCreate {
  name: string;
  width_ft?: number;
  height_ft?: number;
  edge_left?: number;
  edge_right?: number;
  edge_bottom?: number;
  edge_top?: number;
  photo?: File;
}

export interface SizeCreateResponse {
  id: string;
  layout_id: string;
  name: string;
}
