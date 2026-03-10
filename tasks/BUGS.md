# Bug Log

## Fixed

### BUG-001: `createLayout` frontend call missing `image_edges`
**Discovered during**: Task 2 (types + API layer)
**Symptom**: The backend `POST /layouts` endpoint required `image_edges` as a Form field, but the frontend API client (`src/api/layouts.ts`) never appended it to the FormData. This would have caused a 422 Unprocessable Entity error on every layout creation attempt.
**Fix**: Added `image_edges` to the `LayoutCreate` TypeScript type and updated the `createLayout` API function to append it. Also updated `new.tsx` submit to pass `image_edges: [0, widthFt, 0, heightFt]` (done in Task 5).
**Status**: Fixed in Tasks 2 + 5.
