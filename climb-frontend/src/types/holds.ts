export type Tag = "pinch" | "flat";

export interface HoldDetail {
  hold_index: number;
  x: number;
  y: number;
  pull_x: number | null;
  pull_y: number | null;
  useability: number | null;
  is_foot: boolean;
  tags: Tag[];
}

export type HoldMode = "add" | "remove" | "select" | "edit";

export interface EnabledFeatures {
  direction: boolean;
  useability: boolean;
  footholds: boolean;
  tags: boolean;
}

export type FeatureLabel = "direction" | "useability" | "footholds" | "tags";
