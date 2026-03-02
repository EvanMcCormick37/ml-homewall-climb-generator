/**
 * Types for climb-related data structures
 *
 * These types mirror the Pydantic schemas in climb-api/app/schemas/climbs.py
 */

export type ClimbSortBy = "date" | "name" | "grade" | "ascents";

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
  holdset: Holdset;
  grade: number | null;
  quality: number | null;
  ascents: number;
  setter_name: string | null;
  setter_id: string | null;
  tags: string[] | null;
  created_at: Date;
}

export interface ClimbCreate {
  name: string;
  holdset: Holdset;
  angle: number;
  scale: string | null;
  grade: string | null;
  setter_name: string;
  setter_id: string;
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
  gradeScale: string;
  minGrade: string;
  maxGrade: string;
  includeProjects: boolean;
  setterName: string;
  nameIncludes?: string;
  holdsInclude?: number[];
  tagsInclude?: string[];
  after: string;
  sortBy: ClimbSortBy;
  descending: boolean;
  limit?: number;
  offset?: number;
}

export const DEFAULT_CLIMB_FILTERS: ClimbFilters = {
  gradeScale: "v_grade",
  minGrade: "V0-",
  maxGrade: "V16",
  includeProjects: true,
  setterName: "",
  after: "",
  sortBy: "date",
  descending: true,
};
