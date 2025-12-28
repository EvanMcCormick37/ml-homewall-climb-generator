/**
 * Types for climb-related data structures
 *
 * These types mirror the Pydantic schemas in climb-api/app/schemas/climbs.py
 */

export type ClimbSortBy = "date" | "name" | "grade" | "ticks" | "num_moves";

export interface Holdset {
  start: number[];
  finish: number[];
  hand: number[];
  foot: number[];
}

export interface Climb {
  id: string;
  wall_id: string;
  angle: string | null;
  name: string | null;
  grade: number | null;
  setter: string | null;
  holds: Holdset;
  tags: string[] | null;
  num_moves: number;
  created_at: string;
}

export interface ClimbCreate {
  name: string;
  holds: Holdset;
  grade: number | null;
  setter: string | null;
  tags: string[] | null;
}

export interface ClimbListResponse {
  climbs: Climb[];
  total: number;
  limit: number;
  offset: number;
}

export interface ClimbCreateResponse {
  id: string;
}

export interface ClimbDeleteResponse {
  id: string;
}

export interface ClimbFilters {
  grade_range?: [number, number];
  angle?: number;
  include_projects?: boolean;
  setter?: string;
  name_includes?: string;
  holds_include?: number[];
  tags_include?: string[];
  after?: string;
  sort_by?: ClimbSortBy;
  descending?: boolean;
  limit?: number;
  offset?: number;
}
