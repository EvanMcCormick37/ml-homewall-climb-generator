/**
 * Types for climb-related data structures
 *
 * These types mirror the Pydantic schemas in climb-api/app/schemas/climbs.py
 */

export type ClimbSortBy = "date" | "name" | "grade" | "ticks" | "num_moves";

export interface Climb {
  id: string;
  wall_id: string;
  name: string | null;
  grade: number | null;
  setter: string | null;
  sequence: number[][]; // [[lh_hold_id, rh_hold_id], ...]
  tags: string[] | null;
  num_moves: number;
  created_at: string;
}

export interface Holdset {
  start: number[];
  finish: number[];
  hand: number[];
  foot: number[];
}

export interface ClimbCreate {
  name: string;
  holds: Holdset;
  grade?: number | null;
  setter?: string | null;
  tags?: string[] | null;
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

/**
 * Convert numeric grade (0-180) to V-grade string
 * Grade format: V0 = 0-9, V1 = 10-19, etc.
 * Decimal indicates +/- (e.g., V3- = 27, V3 = 30, V3+ = 33)
 */
export function gradeToString(grade: number | null): string {
  if (grade === null || grade === undefined) return "Project";

  const vGrade = Math.floor(grade / 10);
  const decimal = grade % 10;

  let suffix = "";
  if (decimal <= 3) suffix = "-";
  else if (decimal >= 7) suffix = "+";

  return `V${vGrade}${suffix}`;
}

/**
 * Get a color for a given grade
 */
export function gradeToColor(grade: number | null): string {
  if (grade === null) return "#6b7280"; // gray for projects

  const vGrade = Math.floor(grade / 10);

  // Color gradient from green (easy) to red (hard)
  const colors = [
    "#22c55e", // V0 - green
    "#84cc16", // V1 - lime
    "#eab308", // V2 - yellow
    "#f97316", // V3 - orange
    "#ef4444", // V4 - red
    "#dc2626", // V5 - red-600
    "#b91c1c", // V6 - red-700
    "#991b1b", // V7 - red-800
    "#7f1d1d", // V8 - red-900
    "#581c87", // V9+ - purple
  ];

  return colors[Math.min(vGrade, colors.length - 1)];
}
