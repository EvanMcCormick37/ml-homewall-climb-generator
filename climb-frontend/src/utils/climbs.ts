const GRADE_TO_DIFF = {
  font: {
    "4a": 10,
    "4b": 11,
    "4c": 12,
    "5a": 13,
    "5b": 14,
    "5c": 15,
    "6a": 16,
    "6a+": 17,
    "6b": 18,
    "6b+": 19,
    "6c": 20,
    "6c+": 21,
    "7a": 22,
    "7a+": 23,
    "7b": 24,
    "7b+": 25,
    "7c": 26,
    "7c+": 27,
    "8a": 28,
    "8a+": 29,
    "8b": 30,
    "8b+": 31,
    "8c": 32,
    "8c+": 33,
  },
  v_grade: {
    "V0-": 10,
    V0: 11,
    "V0+": 12,
    V1: 13,
    "V1+": 14,
    V2: 15,
    V3: 16,
    "V3+": 17,
    V4: 18,
    "V4+": 19,
    V5: 20,
    "V5+": 21,
    V6: 22,
    "V6+": 22.5,
    V7: 23,
    "V7+": 23.5,
    V8: 24,
    "V8+": 25,
    V9: 26,
    "V9+": 26.5,
    V10: 27,
    "V10+": 27.5,
    V11: 28,
    "V11+": 28.5,
    V12: 29,
    "V12+": 29.5,
    V13: 30,
    "V13+": 30.5,
    V14: 31,
    "V14+": 31.5,
    V15: 32,
    "V15+": 32.5,
    V16: 33,
  },
} as const;

export type GradeScale = keyof typeof GRADE_TO_DIFF;

/**
 * Convert a numeric difficulty value to its grade string by inverse lookup.
 * Finds the closest matching grade if the value falls between two entries.
 */
export function gradeToString(
  grade: number | null,
  gradeScale: GradeScale = "v_grade",
): string {
  if (grade === null || grade === undefined) return "Project";

  const entries = Object.entries(GRADE_TO_DIFF[gradeScale]) as [
    string,
    number,
  ][];

  return entries.reduce(
    (closest, [label, diff]) =>
      Math.abs(diff - grade) <
      Math.abs(
        GRADE_TO_DIFF[gradeScale][
          closest as keyof (typeof GRADE_TO_DIFF)[typeof gradeScale]
        ] - grade,
      )
        ? label
        : closest,
    entries[0][0],
  );
}

/**
 * Get a color for a given grade
 */
export function gradeToColor(grade: number | null): string {
  if (grade === null) return "#0e0e0eff"; // dark for projects

  const vGrade = Math.floor((grade - 10) / 2.3);

  // Color gradient from green (easy) to purple (hard)
  const colors = [
    "#22c55e", // V0
    "#00c717ff", // V1
    "#4acc16ff", // V2
    "#68cc16ff", // V3
    "#b9ea08ff", // V4
    "#d9ff00ff", // V5
    "#e8dc00ff", // V6
    "#ffee00ff", // V7
    "#e6b000ff", // V8
    "#dc6f26ff", // V9
    "#dc3826ff", // V10
    "#b91c1c", // V11
    "#991b2cff", // V12
    "#8a042cff", // V13
    "#8b0071ff", // V14
    "#79007dff", // V15
    "#6c007cff", // V16
    "#3e0075ff", // V17
  ];

  return colors[vGrade];
}
