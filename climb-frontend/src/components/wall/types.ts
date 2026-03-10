import type { Holdset } from "@/types";

// ─── Hold categories ─────────────────────────────────────────────────────────

export type HoldCategory = "hand" | "foot" | "start" | "finish";

export const CATEGORY_ORDER: HoldCategory[] = ["hand", "foot", "start", "finish"];

export const CATEGORY_COLORS: Record<HoldCategory, string> = {
  hand: "#3b82f6",
  foot: "#a855f7",
  start: "#22c55e",
  finish: "#ffea00",
};

export const CATEGORY_LABELS: Record<HoldCategory, string> = {
  hand: "Hand",
  foot: "Foot",
  start: "Start",
  finish: "Finish",
};

export const HOLD_STROKE_COLOR = "#00b679";

// ─── Display settings ────────────────────────────────────────────────────────

export type ColorMode = "role" | "uniform";

export interface DisplaySettings {
  scale: number;
  colorMode: ColorMode;
  uniformColor: string;
  categoryColors: Record<HoldCategory, string>;
  opacity: number;
  filled: boolean;
}

export const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  scale: 1.0,
  colorMode: "role",
  uniformColor: HOLD_STROKE_COLOR,
  categoryColors: { ...CATEGORY_COLORS },
  opacity: 0.6,
  filled: true,
};

// ─── Named holdset (a climb in the generate/edit UI) ─────────────────────────

export interface NamedHoldset {
  name: string;
  holdset: Holdset;
  grade: string;
  angle: string;
}
