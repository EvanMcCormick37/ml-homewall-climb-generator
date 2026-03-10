/**
 * Perspective (homography) transform utilities.
 *
 * A homography maps one plane to another via a 3×3 matrix H. Points are
 * represented in homogeneous coordinates so that the transform is linear.
 *
 * Usage:
 *   const H    = computeHomography(srcPts, dstPts);   // 4 point pairs → matrix
 *   const Hinv = invertMat3(H);
 *   const dst  = applyHomography(H,    [srcX, srcY]);
 *   const src  = applyHomography(Hinv, [dstX, dstY]);
 */

export type Point2D = [number, number];
export type Mat3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

// ─── Gaussian elimination solver ──────────────────────────────────────────────

/**
 * Solve the linear system Ax = b (in-place via Gaussian elimination with
 * partial pivoting). Returns the solution vector x, or throws if singular.
 * A is modified in-place; b is not.
 */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  // Augmented matrix [A | b]
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) throw new Error("Singular matrix in homography solve");

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / pivot;
      for (let k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k];
      }
    }
  }

  // Back-substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= M[i][j] * x[j];
    }
    x[i] /= M[i][i];
  }
  return x;
}

// ─── Homography computation (DLT) ─────────────────────────────────────────────

/**
 * Compute the 3×3 homography matrix H (with h[2][2] = 1) such that for each
 * correspondence (srcPts[i], dstPts[i]):
 *
 *   [dx, dy, 1]^T ∝ H · [sx, sy, 1]^T
 *
 * Requires exactly 4 point pairs (the minimum for a full homography).
 *
 * Algorithm: Direct Linear Transform (DLT).
 * Each point pair produces 2 rows in the 8×8 linear system.
 */
export function computeHomography(srcPts: Point2D[], dstPts: Point2D[]): Mat3 {
  if (srcPts.length !== 4 || dstPts.length !== 4) {
    throw new Error("computeHomography requires exactly 4 point correspondences");
  }

  // Build 8×8 matrix A and rhs b from the DLT equations.
  // For each pair (sx, sy) → (dx, dy):
  //   -sx·h00 - sy·h01 - h02                    + dx·sx·h20 + dx·sy·h21 = -dx
  //                    - sx·h10 - sy·h11 - h12   + dy·sx·h20 + dy·sy·h21 = -dy
  // (where h22 = 1 is fixed, so h20 and h21 appear in rhs adjustment)
  const rows: number[][] = [];
  const rhs: number[] = [];

  for (let i = 0; i < 4; i++) {
    const [sx, sy] = srcPts[i];
    const [dx, dy] = dstPts[i];

    // Row for x equation: h = [h00,h01,h02, h10,h11,h12, h20,h21]
    rows.push([-sx, -sy, -1,   0,   0,  0,  dx * sx, dx * sy]);
    rhs.push(-dx);

    // Row for y equation
    rows.push([  0,   0,  0, -sx, -sy, -1,  dy * sx, dy * sy]);
    rhs.push(-dy);
  }

  const h = solveLinear(rows, rhs);
  // h = [h00, h01, h02, h10, h11, h12, h20, h21], h22 = 1

  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1.0 ],
  ];
}

// ─── Apply homography ──────────────────────────────────────────────────────────

/**
 * Apply a 3×3 homography matrix to a 2D point.
 * Uses homogeneous coordinates with perspective division.
 */
export function applyHomography(H: Mat3, pt: Point2D): Point2D {
  const [x, y] = pt;
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

// ─── 3×3 matrix inversion ──────────────────────────────────────────────────────

/**
 * Invert a 3×3 matrix using the adjugate (cofactor transpose) method.
 * Throws if the matrix is singular.
 */
export function invertMat3(M: Mat3): Mat3 {
  const [[a, b, c], [d, e, f], [g, h, k]] = M;

  const det =
    a * (e * k - f * h) -
    b * (d * k - f * g) +
    c * (d * h - e * g);

  if (Math.abs(det) < 1e-12) throw new Error("invertMat3: singular matrix");

  const inv = 1 / det;
  return [
    [ (e * k - f * h) * inv, -(b * k - c * h) * inv,  (b * f - c * e) * inv],
    [-(d * k - f * g) * inv,  (a * k - c * g) * inv, -(a * f - c * d) * inv],
    [ (d * h - e * g) * inv, -(a * h - b * g) * inv,  (a * e - b * d) * inv],
  ];
}
