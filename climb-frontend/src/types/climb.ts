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
  angle: number;
  name: string;
  grade: number | null;
  setter_name: string | null;
  holdset: Holdset;
  tags: string[] | null;
  ascents: number;
  created_at: Date;
}

export interface ClimbCreate {
  name: string;
  holdset: Holdset;
  angle: number;
  grade: number | null;
  setter_name: string | null;
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
  angle?: number;
  grade_range?: [number, number];
  include_projects?: boolean;
  setter_name?: string;
  name_includes?: string;
  holds_include?: number[];
  tags_include?: string[];
  after?: string;
  sort_by?: ClimbSortBy;
  descending?: boolean;
  limit?: number;
  offset?: number;
}
