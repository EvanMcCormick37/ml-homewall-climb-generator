export const GRADE_TO_DIFF = {
  font: {
    "4a": 10, "4b": 11, "4c": 12,
    "5a": 13, "5b": 14, "5c": 15,
    "6a": 16, "6a+": 17, "6b": 18, "6b+": 19,
    "6c": 20, "6c+": 21,
    "7a": 22, "7a+": 23, "7b": 24, "7b+": 25,
    "7c": 26, "7c+": 27,
    "8a": 28, "8a+": 29, "8b": 30, "8b+": 31,
    "8c": 32, "8c+": 33,
  },
  v_grade: {
    "V0-": 10, "V0": 11, "V0+": 12,
    "V1": 13, "V1+": 14, "V2": 15, "V2+": 15.5,
    "V3": 16, "V3+": 17, "V4": 18, "V4+": 19,
    "V5": 20, "V5+": 21, "V6": 22, "V6+": 22.5,
    "V7": 23, "V7+": 23.5, "V8": 24, "V8+": 25,
    "V9": 26, "V9+": 26.5, "V10": 27, "V10+": 27.5,
    "V11": 28, "V11+": 28.5, "V12": 29, "V12+": 29.5,
    "V13": 30, "V13+": 30.5, "V14": 31, "V14+": 31.5,
    "V15": 32, "V15+": 32.5, "V16": 33,
  },
} as const;

export type GradeScale = keyof typeof GRADE_TO_DIFF;

/** Ordered list of valid grade strings for a given scale. */
export function gradeOptions(scale: GradeScale): string[] {
  return Object.keys(GRADE_TO_DIFF[scale]);
}

/** Convert a grade string to its difficulty float. */
export function gradeToDifficulty(grade: string, scale: GradeScale): number {
  return (GRADE_TO_DIFF[scale] as Record<string, number>)[grade];
}

/**
 * Convert a difficulty float to its nearest grade string.
 * Finds the closest matching grade when the value falls between entries.
 */
export function gradeToString(
  difficulty: number | null,
  gradeScale: GradeScale = "v_grade",
): string {
  if (difficulty === null || difficulty === undefined) return "Project";

  const entries = Object.entries(GRADE_TO_DIFF[gradeScale]) as [string, number][];
  return entries.reduce(
    (closest, [label, diff]) =>
      Math.abs(diff - difficulty) <
      Math.abs(
        (GRADE_TO_DIFF[gradeScale] as Record<string, number>)[closest] - difficulty,
      )
        ? label
        : closest,
    entries[0][0],
  );
}

/** Get a color for a given difficulty value. */
export function gradeToColor(difficulty: number | null): string {
  if (difficulty === null) return "#0e0e0eff";
  const vGrade = Math.floor((difficulty - 10) / 2.3);
  const colors = [
    "rgb(0, 235, 25)",
    "rgb(0, 175, 125)",
    "rgb(0, 200, 200)",
    "rgb(0, 100, 225)",
    "rgb(0, 0, 255)",
    "rgb(255, 255, 0)",
    "rgb(255, 205, 0)",
    "rgb(255, 180, 0)",
    "rgb(255, 140, 0)",
    "rgb(255, 100, 0)",
    "rgb(220, 55, 25)",
    "rgb(190, 25, 25)",
    "rgb(165, 0, 25)",
    "rgb(165, 0, 55)",
    "rgb(135, 0, 90)",
    "rgb(120, 0, 125)",
    "rgb(100, 0, 100)",
    "rgb(50, 0, 75)",
  ];
  return colors[Math.max(0, Math.min(colors.length - 1, vGrade))];
}
