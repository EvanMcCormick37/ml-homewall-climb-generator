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
  if (grade === null) return "#0e0e0eff"; // dark for projects

  const vGrade = Math.floor(grade / 10);

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
