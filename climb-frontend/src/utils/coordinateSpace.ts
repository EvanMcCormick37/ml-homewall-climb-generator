/**
 * Pure coordinate-space transforms shared by the wall canvas and hold editor.
 *
 * Holds are stored in feet (origin: bottom-left of wall).
 * Canvas pixels use origin: top-left of image, Y increases downward.
 *
 * When homography corners are available, perspective transform is used;
 * otherwise falls back to affine mapping via image_edges.
 */
import {
  applyHomography,
  computeHomography,
  invertMat3,
  type Mat3,
  type Point2D,
} from "@/utils/homography";

export interface Dimensions {
  width: number;
  height: number;
}

/**
 * Compute H (pixel→feet) and Hinv (feet→pixel) from trapezoid corner data.
 * Returns null matrices when corners are absent or invalid.
 */
export function buildWallHomography(
  homographySrcCorners: number[] | null | undefined,
  imageDimensions: Dimensions,
  wallDimensions: Dimensions,
): { H: Mat3 | null; Hinv: Mat3 | null } {
  if (
    !homographySrcCorners ||
    homographySrcCorners.length !== 8 ||
    imageDimensions.width === 0 ||
    imageDimensions.height === 0
  ) {
    return { H: null, Hinv: null };
  }

  const { width: iW, height: iH } = imageDimensions;
  const { width: W, height: Hft } = wallDimensions;

  // Source: corner positions in pixel space (TL, TR, BL, BR)
  const srcPts: Point2D[] = [
    [homographySrcCorners[0] * iW, homographySrcCorners[1] * iH],
    [homographySrcCorners[2] * iW, homographySrcCorners[3] * iH],
    [homographySrcCorners[4] * iW, homographySrcCorners[5] * iH],
    [homographySrcCorners[6] * iW, homographySrcCorners[7] * iH],
  ];
  // Destination: wall-coordinate space in feet (TL, TR, BL, BR)
  const dstPts: Point2D[] = [
    [0, Hft],
    [W, Hft],
    [0, 0],
    [W, 0],
  ];

  try {
    const H = computeHomography(srcPts, dstPts);
    return { H, Hinv: invertMat3(H) };
  } catch {
    return { H: null, Hinv: null };
  }
}

function resolveEdges(
  imageEdges: [number, number, number, number] | null | undefined,
  wallDimensions: Dimensions,
): [number, number, number, number] {
  return imageEdges ?? [0, wallDimensions.width, 0, wallDimensions.height];
}

/** Convert a hold (feet) to canvas pixel coordinates. */
export function holdToPixel(
  hold: { x: number; y: number },
  Hinv: Mat3 | null,
  imageEdges: [number, number, number, number] | null | undefined,
  imageDimensions: Dimensions,
  wallDimensions: Dimensions,
): { x: number; y: number } {
  if (Hinv) {
    const [px, py] = applyHomography(Hinv, [hold.x, hold.y]);
    return { x: px, y: py };
  }
  const [imgL, imgR, imgB, imgT] = resolveEdges(imageEdges, wallDimensions);
  return {
    x: ((hold.x - imgL) / (imgR - imgL)) * imageDimensions.width,
    y: ((imgT - hold.y) / (imgT - imgB)) * imageDimensions.height,
  };
}

/** Convert canvas pixel coordinates to wall feet. */
export function pixelToFeet(
  pixelX: number,
  pixelY: number,
  H: Mat3 | null,
  imageEdges: [number, number, number, number] | null | undefined,
  imageDimensions: Dimensions,
  wallDimensions: Dimensions,
): { x: number; y: number } {
  if (H) {
    const [xFeet, yFeet] = applyHomography(H, [pixelX, pixelY]);
    return { x: xFeet, y: yFeet };
  }
  const [imgL, imgR, imgB, imgT] = resolveEdges(imageEdges, wallDimensions);
  return {
    x: imgL + (pixelX / imageDimensions.width) * (imgR - imgL),
    y: imgB + ((imageDimensions.height - pixelY) / imageDimensions.height) * (imgT - imgB),
  };
}
