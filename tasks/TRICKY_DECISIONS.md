# Tricky Decisions

## Decision 1: When to drop the `walls` table

**Question:** The Phase 1 roadmap says "old `walls` table dropped" as the deliverable.
But `wall_service.py`, `routers/walls.py`, and `auth.py` all query the `walls` table
directly. Dropping it in Phase 1 would break the entire API before Phase 2 services
are updated.

**Decision:** Keep the `walls` table alive through Phases 1–5. It gets dropped in
Phase 6 cleanup, once `layout_service.py` is fully in place and all references to
`wall_service.py` have been removed. The migration script in Phase 1 populates the
new `layouts` + `sizes` tables from `walls`, but does not drop `walls`.

Similarly, `wall_id` in `holds` and `climbs` is kept and `layout_id` is added as
a new column (populated with the same value). Services use `wall_id` until Phase 2
swaps them to `layout_id`. Both columns coexist through Phase 5.

**Status:** Decided. Proceeding.

---

## Decision 2: `layout_id` vs keeping `wall_id` in `Climb` schema

**Question:** The `Climb` Pydantic schema and frontend `Climb` type both use `wall_id`.
Phase 2 will introduce a `layout_id` field. During the transition, should the
response include both fields, or rename immediately?

**Decision:** In Phase 2, rename `wall_id` → `layout_id` in the `Climb` schema and
add `layout_id` to the DB response. Keep `wall_id` as an alias field (set to the
same value) for backward compat until Phase 4 frontend types are updated. Remove
the alias in Phase 6.

**Status:** Pending (Phase 2 decision).

---

## Decision 4: Size-aware hold filtering in the DDPM generator

**Question:** The DDPM generator (`generation_utils.py`) loads hold manifolds at
startup, keyed by `wall_id`. Holds are loaded from the DB with `SELECT ... FROM
holds` grouped by `wall_id`. To support size-aware generation (filtering holds to
those within the size's edge bounds), we'd need to either:
  a) Create separate manifold entries per size at startup (expensive, combinatorial)
  b) Pass a hold-mask at generation time and modify the projection logic

**Decision:** Defer this for now. The `generation_service.generate_climbs` function
accepts `size_id` as a parameter (plumbing is in place), but does not apply it to
manifold filtering yet. For the current use-case, generates use the full layout
holdset. A "size_id" flag can gate a future enhancement. Logged as a known limitation.

**Status:** Deferred (non-blocking for Phase 2).

---

## Decision 3: Photo storage path for migrated sizes

**Question:** Currently photos live at `data/walls/{wall_id}/photo.jpg`. After
migration, each wall becomes a layout + one size. Where should the photo be
referenced?

**Decision:** Keep photos in their current location during migration. The `sizes`
table stores `photo_path` as just the filename (e.g., `"photo.jpg"`), and the
lookup resolves it as `settings.WALLS_DIR / layout_id / photo_path` — exactly
the same as the current walls lookup. New sizes (created post-migration) will
use a new path convention: `settings.LAYOUTS_DIR / layout_id / sizes / size_id / photo.jpg`.
A LAYOUTS_DIR config key is added pointing to `data/layouts/`.

**Status:** Decided. Proceeding.
