# BetaZero Roadmap: Trapezoid Crop + Use As-Is Feature

## Feature Overview

Add two new options to the "Crop" step of the new layout wizard (`/layouts/new`):

1. **Use As-Is** — Skip cropping entirely. Upload the original image unchanged. The image is treated as mapping directly to the full wall dimensions (`image_edges = [0, W, 0, H]`). No homography.

2. **Trapezoidal Mesh** — The user drags 4 corner handles on the photo to mark where the physical corners of the wall appear in the image. These 4 source points (normalized 0-1 image coords) are stored as `homography_src_corners` and define a perspective transform from pixel space to wall-feet space. The original (uncropped) image is uploaded. All hold placement and coordinate conversion throughout the app uses this homography when present.

The existing **Rectangular Crop** mode is preserved as the default.

---

## Data Model

`homography_src_corners`: 8 floats in order `[tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]`, all normalized to [0, 1] image space. These map to wall-feet:
- TL → `(0, H_ft)` (top-left in image = top-left of wall = max height)
- TR → `(W_ft, H_ft)`
- BL → `(0, 0)`
- BR → `(W_ft, 0)`

Null when not using trapezoid mode.

---

## Coordinate System Note

Wall coords: origin (0,0) = bottom-left, y increases upward.
Image/pixel coords: origin (0,0) = top-left, y increases downward.
The existing affine mapping in `useHolds.ts` and `WallCanvas.tsx` already handles this flip.
The homography implementation must be consistent with this convention.

---

## Task List

---

### TASK 1 — Backend: Add `homography_src_corners` to data layer
STATUS: COMPLETED

**What**: Add the new field to the SQLite schema, Pydantic schemas, service, and conversion utilities.

**Files**:
- `climb-backend/app/database.py` — add migration: `_add_column_if_missing(conn, "layouts", "homography_src_corners", "TEXT DEFAULT NULL")`; also add it to all `SELECT` queries in `init_db` area (it's already done via ALTER if missing)
- `climb-backend/app/schemas/layouts.py` — add `homography_src_corners: list[float] | None = None` to `LayoutMetadata`, `LayoutCreate`, `LayoutEdit`
- `climb-backend/app/services/utils/conversion_utils.py` — update `_row_to_layout_metadata` to parse `homography_src_corners` (JSON decode if not null)
- `climb-backend/app/services/layout_service.py` — update `create_layout` INSERT and SELECT queries to include the column; update `put_layout` to support updating it; update all SELECT column lists to include `homography_src_corners`
- `climb-backend/app/routers/layouts.py` — verify the field flows through (likely automatic via Pydantic)

**Acceptance**: Backend returns `homography_src_corners` in layout API responses (null for existing layouts). `createLayout` with `homography_src_corners` stores and returns it.

---

### TASK 2 — Frontend: Types + API layer
STATUS: COMPLETED

**What**: Update TypeScript types and the API client to carry `homography_src_corners`.

**Files**:
- `climb-frontend/src/types/layout.ts`:
  - Add `homography_src_corners?: number[] | null` to `LayoutMetadata`
  - Add `homography_src_corners?: number[] | null` to `LayoutCreate`
  - Add `homography_src_corners?: number[] | null` to `LayoutUpdate`
- `climb-frontend/src/api/layouts.ts` — verify `createLayout` and `updateLayout` accept and send the field (likely automatic since they spread the object)

**Acceptance**: TypeScript compiles cleanly (`npx tsc --noEmit`). The field appears in the typed response from `getLayout()`.

---

### TASK 3 — Frontend: Homography math utilities
STATUS: COMPLETED

**What**: Pure TypeScript implementation of perspective transform math. No external library.

**New file**: `climb-frontend/src/utils/homography.ts`

**Exports**:
```typescript
type Point2D = [number, number];
type Mat3 = number[][]; // 3x3

// Solve for the 3x3 homography matrix H such that dst_i ≈ H * src_i (in homogeneous coords)
// Requires exactly 4 point pairs.
function computeHomography(srcPts: Point2D[], dstPts: Point2D[]): Mat3

// Apply homography to a point (handles homogeneous division)
function applyHomography(H: Mat3, pt: Point2D): Point2D

// Invert a 3x3 matrix (using cofactor / adjugate method)
function invertMat3(H: Mat3): Mat3
```

**Algorithm**: The standard DLT (Direct Linear Transform) method — each point correspondence produces 2 rows in the 8×8 linear system Ah=0. Solve via Gaussian elimination for the 8 unknowns of H (normalized so h[2][2]=1).

**Acceptance**: Unit-testable in isolation. A round-trip test: `applyHomography(H, src) ≈ dst` for all 4 correspondences.

---

### TASK 4 — Frontend: `TrapezoidCropper` component
STATUS: COMPLETED

**What**: New UI component for the trapezoid corner selection step.

**New file**: `climb-frontend/src/components/TrapezoidCropper.tsx`

**Interface**:
```typescript
interface TrapezoidCropperProps {
  imageUrl: string;
  corners: [number, number, number, number, number, number, number, number]; // [tlx,tly, trx,try, blx,bly, brx,bry] normalized
  onChange: (corners: [number, number, number, number, number, number, number, number]) => void;
}
```

**Implementation details**:
- Renders the image in a `position: relative` container
- Overlays an SVG at full size (pointer-events layer for handles)
- Draws a polygon outline connecting the 4 corners
- Draws a darkened fill outside the quad (SVG clip-path or fill-rule="evenodd" with an outer rect)
- 4 draggable handles (cyan circles, labeled "TL", "TR", "BL", "BR") at the corner positions
- Handle drag: mouse/touch events, clamp to [0, 1] bounds
- Default initial corners: `[0.05, 0.05,  0.95, 0.05,  0.05, 0.95,  0.95, 0.95]` (slight inset)

**Acceptance**: Handles drag correctly, corners update in real time, visual feedback clear.

---

### TASK 5 — Frontend: Update `new.tsx` crop step
STATUS: COMPLETED

**What**: Modify the "Crop" step in the new layout wizard to support 3 modes.

**File**: `climb-frontend/src/routes/layouts/new.tsx`

**Changes**:
- Add state: `cropMode: "none" | "rect" | "trapezoid"` (default `"rect"`)
- Add state: `trapCorners: [number, number, number, number, number, number, number, number]` (8 floats, default inset from corners)
- Add 3-button mode selector at the top of the crop step UI (styled as a segmented control, BetaZero theme)
- Conditionally render:
  - `"rect"`: existing `<ImageCropper>` (no change)
  - `"trapezoid"`: new `<TrapezoidCropper>` with `imageUrl` from `useImageCrop` and `corners`/`onChange`
  - `"none"`: show static preview of the uploaded image with a note ("Image will be used as uploaded")
- Update the final submit (Details step) to:
  - `"none"`: upload original file (not `getCroppedImage()`) — create a Blob from the raw file; pass `homography_src_corners: null`
  - `"rect"`: existing behavior — upload cropped blob; pass `homography_src_corners: null`
  - `"trapezoid"`: upload original file; pass `homography_src_corners: trapCorners` in `createLayout` call
- In all cases `image_edges` stays `[0, W, 0, H]` (the uploaded image always represents the full wall, whether cropped to it or mapped via homography)

**Acceptance**: All 3 modes flow through to completion. New layout created with correct `homography_src_corners` (null for none/rect, 8-float array for trapezoid).

---

### TASK 6 — Frontend: Update `useHolds` for homography
STATUS: COMPLETED

**What**: Update the coordinate conversion hook to use a perspective transform when `homography_src_corners` is provided.

**File**: `climb-frontend/src/hooks/useHolds.ts`

**Changes**:
- Add 4th parameter: `homographySrcCorners?: number[] | null`
- When `homographySrcCorners` is provided (and has 8 elements):
  - Use `useMemo` to compute `H` (pixel→feet) and `H_inv` (feet→pixel) from `homography.ts` utilities
  - Source points: the 4 corners as pixel coordinates = `corners[i] * imgWidth` (or imgHeight)
  - Destination points: `(0, H_ft), (W_ft, H_ft), (0, 0), (W_ft, 0)` for TL, TR, BL, BR
  - `toFeetCoords(px, py)`: apply H to `[px, py]`
  - `toPixelCoords(hold)`: apply H_inv to `[hold.x, hold.y]`
- When null/undefined: existing linear `image_edges` path unchanged

**Acceptance**: Holds placed on a trapezoid-mapped wall appear at the correct visual position in the hold editor. Round-trip: place hold at a known pixel position → save → reload → hold renders at same pixel position.

---

### TASK 7 — Frontend: Wire `holds.tsx`
STATUS: COMPLETED

**What**: Pass `homography_src_corners` from the loaded layout to `useHolds`.

**File**: `climb-frontend/src/routes/$layoutId/holds.tsx`

**Changes**:
- Extract `homographySrcCorners = layout.metadata.homography_src_corners ?? null`
- Pass as 4th argument to `useHolds(imageDimensions, wallDimensions, imageEdges, homographySrcCorners)`

**Acceptance**: Holds editor correctly positions holds for both rect-crop and trapezoid layouts.

---

### TASK 8 — Frontend: Update `WallCanvas` for homography
STATUS: COMPLETED

**What**: The climb viewer/generator uses `WallCanvas`, which has its own pixel↔feet mapping for rendering holds on the image. Update it to support homography.

**File**: `climb-frontend/src/components/wall/WallCanvas.tsx`

**Changes**:
- Add prop: `homographySrcCorners?: number[] | null`
- When present, use `computeHomography` + `applyHomography` (from `homography.ts`) for `toPixelCoords` inside the rendering loop — replacing the existing linear formula
- This affects all hold rendering, size overlay, and click-hit detection

**File**: `climb-frontend/src/routes/$layoutId/set.tsx` and `$layoutId/view.tsx`
- Pass `homographySrcCorners={layout.metadata.homography_src_corners}` to `<WallCanvas>`

**Acceptance**: Holds render at the correct position on the wall photo in the climb viewer/generator for trapezoid-mapped layouts.

---

## Known Limitations / Deferred Work

See `tasks/TRICKY_DECISIONS.md` for full notes. Summary:
- **`sizes.tsx` visual overlay**: The edge-crop overlay on the sizes page is not geometrically accurate for trapezoid layouts (since the overlay is axis-aligned in image space, not wall-feet space). Sizes still work functionally (hold filtering uses wall-feet coords). Fixing the overlay to show a perspective-correct region is deferred.
- **Image perspective warp**: The stored photo for trapezoid mode is the original (unperspective-corrected) image. A future enhancement could warp the image to a rectangle before uploading, simplifying the rest of the system, but this requires a triangle-mesh canvas approach and is non-trivial.

---

## Organization instructions:

After reading this Roadmap, take the first task shown here, memorize it, mark it as STATUS: WORKING, and create a more detailed version of it in CURRENT_TASK.md, breaking it up into more detailed sub-tasks. Then, go through the sub-tasks in CURRENT_TASK.md and work through them one-by-one (marking them as INCOMPLETE, WORKING, and COMPLETED in the same way you're doing here). Once all of the subtasks in CURRENT_TASK.md are completed, empty out the CURRENT_TASK.md tasklist, come back here, and mark the task as COMPLETED. Then start the next task on the list, following the same procedure.

## Dealing with bugs:

If you run into trouble or come across a bug while performing this update, write down the bug in BUGS.md. If the bug is something which must be fixed for work to continue, then do your best to fix it, and delete it from BUGS.md when you have fixed it. However, if it isn't necessary to fix it immediately, then just move on to the next task. I will look through the bugs later and work on them.

## What to do if you're unsure of what to do:

This is an autonomous task, and I expect it may take awhile and involve a few nuanced or tricky decisions. Use your best judgement whenever possible. If you must ask for clarification, then ask for clarification. Otherwise, go to TRICKY_DECISIONS.md and write down the question/decision you are having trouble with and your thoughts on the situation. Then, if you feel confident enough to make a call, write down the decision you've made and continue with the work. If you can't make the decision, but you feel like you can continue to do useful work on other parts of this roadmap, then write down the decision, and move on to other work.

## One more thing

Don't commit these changes to git yet. Just leave them as is for now.
