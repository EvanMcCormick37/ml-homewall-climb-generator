## 9. Development Roadmap

### Phase 1 — Database & Migration (backend only, ~1 session) STATUS: COMPLETED

1. Add `layouts` and `sizes` tables to `database.py` → `init_db()`.
2. Write `db_migration.py` script to migrate existing walls → layouts + sizes.
3. Run migration against dev DB. Verify data integrity.
4. Update `holds` table: `wall_id` → `layout_id` (SQLite rename via migration).
5. Update `climbs` table: `wall_id` → `layout_id`, add `size_id` column.

**Deliverable:** DB migrated, all old data intact, old `walls` table dropped.

---

### Phase 2 — Backend: new schemas and services (~1 session) STATUS: COMPLETED

1. Create `app/schemas/layouts.py` — `LayoutMetadata`, `LayoutCreate`, `LayoutDetail`, `LayoutListResponse`.
2. Create `app/schemas/sizes.py` — `SizeMetadata`, `SizeCreate`, `SizeDetail`.
3. Create `app/services/layout_service.py` — port `wall_service.py`, replace `wall_id` with `layout_id`, add size-aware `get_holds(layout_id, size_id=None)`.
4. Create `app/services/size_service.py` — CRUD for sizes, photo management.
5. Update `app/services/generation_service.py` — accept `size_id`, apply edge-bound filtering.
6. Update `app/services/climb_service.py` — `wall_id` → `layout_id`, add `size_id` filter in `get_climbs`.

**Deliverable:** All service logic updated, old `wall_service.py` preserved but deprecated.

---

### Phase 3 — Backend: new API routes (~1 session) STATUS: COMPLETED

1. Create `app/routers/layouts.py` — all `/layouts/...` endpoints.
2. Create `app/routers/sizes.py` — all `/layouts/{layout_id}/sizes/...` endpoints.
3. Update `app/main.py` — register new routers, keep old `/walls/` as alias redirects temporarily.
4. Update `app/test/test_api.py` — update tests to use new routes.
5. Manual smoke test all endpoints.

**Deliverable:** Full API functional on new routes. Old routes still respond (proxied).

---

### Phase 4 — Frontend: types and API client (~1 session) STATUS: COMPLETED

1. Add `src/types/layout.ts` — `LayoutMetadata`, `LayoutDetail`, `SizeMetadata`, `LayoutListResponse`, etc.
2. Update `src/types/wall.ts` — keep for backward compat during transition, mark deprecated.
3. Add `src/api/layouts.ts` — `getLayouts()`, `getLayout(id)`, `createLayout()`, `deleteLayout()`.
4. Add `src/api/sizes.ts` — `createSize()`, `deleteSize()`, `getSizePhotoUrl()`, `uploadSizePhoto()`.
5. Update `src/api/walls.ts` — proxy to layouts.ts during transition.
6. Run `npx tsc --noEmit` — fix all type errors.

**Deliverable:** Frontend compiles with new types and API client.

---

### Phase 5 — Frontend: routes and pages (~2 sessions) STATUS: COMPLETED

1. Update `src/routes/index.tsx` — render `LayoutMetadata[]`, use `getSizePhotoUrl` for card images, link to `/$layoutId` or `/$layoutId/$sizeId/set` for single-size.
2. Add `src/routes/$layoutId/index.tsx` (or `sizes.tsx`) — size picker page for multi-size layouts.
3. Update `src/routes/$wallId/holds.tsx` → `src/routes/$layoutId/holds.tsx` — layout-level, no size context needed.
4. Update `src/routes/$wallId/set.tsx` → `src/routes/$layoutId/$sizeId/set.tsx` — pass `size_id` to generate endpoint; use `getSizePhotoUrl` for the wall image.
5. Update `src/routes/$wallId/view.tsx` → `src/routes/$layoutId/$sizeId/view.tsx` — pass `size_id` to climbs list endpoint.
6. Update `src/routes/walls/new.tsx` → `src/routes/layouts/new.tsx` — collect size info as first size in wizard.
7. Update `src/hooks/useWalls.ts` → `src/hooks/useLayouts.ts`.
8. Add `src/hooks/useSizes.ts`.
9. Run `npx tsc --noEmit`. Fix all errors.

**Deliverable:** Full frontend on new routing, fully functional, all existing wall IDs still work.

---

### Phase 6 — Cleanup (~0.5 session) STATUS: DEFERRED (pending end-to-end validation of layout routes in production)

1. Remove `/api/v1/walls/` alias endpoints from backend.
2. Remove deprecated `src/types/wall.ts` (or reduce to a re-export of layout types).
3. Remove deprecated `src/api/walls.ts` (or reduce to re-exports).
4. Remove deprecated `wall_service.py` and `routers/walls.py`.
5. Final `npx tsc --noEmit`. Final backend test run.
6. Commit and tag.

**Deliverable:** Clean codebase with no dead code, full layouts+sizes support.

---

#### Organization instructions:

After reading this Roadmap, take the first task shown here, memorize it, mark it as STATUS: WORKING, and create a more detailed version of it in CURRENT_TASK.md, breaking it up into more detailed sub-tasks. Then, go through the sub-tasks in CURRENT_TASK.md and work through them one-by-one (marking them as INCOMPLETE, WORKING, and COMPLETED int he same way you're doing here). Once all of the subtasks in CURRENT_TASK.md are completed, empty out the CURRENT_TASK.md tasklist, come back here, and mark the task as COMPLETED. Then start the next task on the list, following the same procedure.

#### Dealing with bugs:

If you run into trouble or come across a bug while performing this update, write down the bug in BUGS.md. If the bug is something which must be fixed for work to continue, then do your best to fix it, and delete it from BUGS.md when you have fixed it. However, if it isn't necessary to fix it immediately, then just move on to the next task. I will look through the bugs later and work on them.

#### What to do if you're unsure of what to do:

This is an autonomous task, and I expect it may take awhile and involve a few nuanced or tricky decisions. Use your best judgement whenever possible. If you must ask for clarification, then ask for clarification. Otherwise, go to TRICKY_DECISIONS.md and write down the question/decision you are having trouble with and your thoughts on the situation. Then, if you feel confident enough to make a call, write down the decision you've made and continue with the work. If you can't make the decision, but you feel like you can continue to do useful work on other parts of this roadmap, then write down the decison, and move on to other work.

#### One more thing (I forgot)

Don't commit these changes to git yet. just leave them as is for now.
