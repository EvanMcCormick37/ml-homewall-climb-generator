/**
 * Types for layout/size data structures.
 *
 * A layout is a unique hold arrangement (replaces the old Wall concept).
 * A size is a physical dimension variant of a layout (different edges/kickboard).
 * Climbs and holds are tied to a layout, not a size.
 * Photos are owned by the layout, not the size.
 */

import type { HoldDetail } from "./holds";

export type Visibility = "public" | "private" | "unlisted";

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
  image_edges: number[] | null; // [left, right, bottom, top] in ft — where each image edge sits in wall coords
  homography_src_corners: number[] | null; // [tlx,tly, trx,try, blx,bly, brx,bry] normalized 0-1 — null for rect/as-is mode
  default_angle: number | null;
  sizes: SizeMetadata[];
  climb_count: number;
  owner_id: string;
  visibility: Visibility;
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
  image_edges: number[]; // [left, right, bottom, top] in ft
  homography_src_corners?: number[] | null; // 8 floats, null when not using trapezoid mode
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

export interface LayoutUpdate {
  name?: string;
  description?: string;
  visibility?: "public" | "private" | "unlisted";
  default_angle?: number | null;
  image_edges?: number[]; // [left, right, bottom, top] in ft
  homography_src_corners?: number[] | null;
}
