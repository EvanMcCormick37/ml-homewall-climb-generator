# Current Task: Phase 4 — Frontend: Types and API Client

## Goal
Add TypeScript types for layouts/sizes and a new API client layer so the
frontend can talk to the new `/api/v1/layouts/...` endpoints. Keep the old
wall types/API intact for now (backward compat until Phase 6).

---

## Sub-tasks

- [x] 1. Create `src/types/layout.ts` — LayoutMetadata, LayoutDetail, SizeMetadata, LayoutListResponse, LayoutCreate, LayoutCreateResponse
- [x] 2. Keep `src/types/wall.ts` as-is (backward compat)
- [x] 3. Create `src/api/layouts.ts` — getLayouts(), getLayout(id), createLayout(), deleteLayout(), setLayoutHolds(), getSizePhotoUrl()
- [x] 4. Create `src/api/sizes.ts` — getSizes(layoutId), createSize(), deleteSize(), uploadSizePhoto()
- [x] 5. Run `npx tsc --noEmit` — 0 errors
- [x] 6. Mark Phase 4 COMPLETED in ROADMAP.md

---

# Current Task: Phase 6 — Cleanup

## Goal
Remove legacy walls API routes from the backend, remove deprecated wall types/API from frontend, and remove deprecated wall_service and routers/walls.py. Final type-check and backend test run.

## Sub-tasks

- [ ] 1. Backend: Remove `/api/v1/walls/` legacy routes from `app/main.py`
- [ ] 2. Backend: Remove (or gut) `app/routers/walls.py`
- [ ] 3. Backend: Remove (or gut) `app/services/wall_service.py`
- [ ] 4. Backend: Run backend tests and verify no regressions
- [ ] 5. Frontend: Remove `src/types/wall.ts` wall-specific types no longer needed (keep HoldDetail, HoldMode, Visibility, EnabledFeatures, FeatureLabel — needed by other code)
- [ ] 6. Frontend: Remove `src/api/walls.ts` (or reduce to stubs)
- [ ] 7. Frontend: Run `npx tsc --noEmit` — fix any errors
- [ ] 8. Mark Phase 6 COMPLETED in ROADMAP.md

---

**NOTE:** Phase 6 is destructive (removing legacy compat). Before running, confirm that the new layout API routes are working in production. The ROADMAP says "Don't commit these changes to git yet" — Phase 6 cleanup should wait until we're ready to cut over.

## Decision: Defer Phase 6

Phase 6 removes backward-compatibility that is still needed while the new routes are being validated. Keeping legacy routes alive allows the app to fall back to the old API if needed. **Deferring Phase 6** until the new layout routes are tested end-to-end in a live environment.

Mark Phase 6 as DEFERRED for now.

## Goal
Update all frontend routes from wall-based to layout-based. Homepage uses layouts. New wizard creates layout+size. Route param renamed from `wallId` → `layoutId`. Holds editor uses new layout API. Set/view pages keep using wall hook via legacy API (still valid since wall_id == layout_id).

## Sub-tasks

- [x] 1. Create `src/hooks/useLayouts.ts` — `useLayouts()` + `useLayout(id)`
- [x] 2. Update `src/hooks/index.ts` — export useLayouts
- [x] 3. Update `src/routes/index.tsx` — use `useLayouts`, `getSizePhotoUrl` for card images, link to `/$layoutId/set`
- [x] 4. Create `src/routes/layouts/new.tsx` — two-step API: createLayout() then createSize() with photo; navigate to `/$layoutId/holds`
- [x] 5. Create `src/routes/$layoutId/holds.tsx` — from `$wallId/holds.tsx`, use `getLayout`, `setLayoutHolds`, extract dims from sizes[0]
- [x] 6. Create `src/routes/$layoutId/set.tsx` — from `$wallId/set.tsx`, rename param `wallId`→`layoutId`, keep `useWall(layoutId)` (legacy compat)
- [x] 7. Create `src/routes/$layoutId/view.tsx` — from `$wallId/view.tsx`, rename param `wallId`→`layoutId`, keep `useWall(layoutId)`
- [x] 8. Delete old `src/routes/$wallId/` files and `src/routes/walls/new.tsx`
- [x] 9. Run `npx tsc --noEmit` — 0 errors
- [x] 10. Mark Phase 5 COMPLETED in ROADMAP.md
