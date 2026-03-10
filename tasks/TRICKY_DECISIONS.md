# Tricky Decisions

## 1. sizes.tsx compatibility with trapezoid (homography) layouts

**Problem**: The `sizes.tsx` page shows a draggable edge-crop overlay on the layout photo. This overlay is axis-aligned in image space (left/right/top/bottom lines). For a trapezoid-mapped layout, the photo is not a direct orthographic projection of the wall, so the overlay lines don't correspond to meaningful physical positions on the wall.

**Decision**: Leave `sizes.tsx` unchanged for now. Sizes still work correctly in feet-space — hold filtering uses wall-feet coordinates, and the size edges are stored and compared in feet. The visual overlay will appear misaligned for trapezoid layouts (it will look like straight lines that don't align with the wall's perspective), but this is a cosmetic issue only. A future improvement could render a perspective-correct overlay using the homography corners.

**Status**: Deferred. Known limitation documented here.

---

## 2. Should the uploaded photo be perspective-corrected for trapezoid mode?

**Problem**: Option A = upload the raw photo + store corner points. Option B = perform a perspective warp in the browser and upload a rectified rectangular image (making the rest of the system work exactly like the rect mode).

**Decision**: Option A. Browser canvas does not natively support perspective warping for arbitrary quads (would require a triangle-mesh approach). The complexity is not justified for the initial implementation. Storing corners and doing the math in JS is simpler, more reversible, and easy to extend later.

---

## 3. Corner ordering convention

**Decision**: Store as `[tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]` in normalized [0,1] image space.
These 4 source points map to wall-feet destinations:
- TL → `(0, H_ft)`
- TR → `(W_ft, H_ft)`
- BL → `(0, 0)`
- BR → `(W_ft, 0)`

This is consistent with the existing image-to-wall y-flip convention in `useHolds.ts`.
