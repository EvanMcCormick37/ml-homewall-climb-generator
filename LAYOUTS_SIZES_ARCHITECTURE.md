# BetaZero: Layouts & Sizes — Architecture & Roadmap

**Author:** Claude (architect review)
**Date:** 2026-03-06
**Branch:** feature-development
**Status:** Proposal — not yet implemented

---

## 1. Current State Assessment

### What exists today

The current system is built around a single `walls` table that conflates several distinct concerns:

```
walls
  id, name, photo_path, num_holds, num_climbs,
  dimensions, angle,
  owner_id, visibility, share_token,
  created_at, updated_at

holds         →  wall_id
climbs        →  wall_id
generate      →  wall_id
```

Every page, every API route, and every ML generation call pivots on `wall_id`. The "wall" is simultaneously the hold arrangement, the physical dimensions, the image, and the browsable unit.

### The problem

This works for one-size-fits-all home walls, but it breaks down when you have commercial boards like the Tension Board 2, where:

- **TB2 Mirror** and **TB2 Spray** are two distinct *hold arrangements* (layouts) for the same product family.
- Each layout comes in multiple *physical sizes* (8×10, 8×12, 12×10, 12×12, etc.).
- A 12×10 TB2 Spray uses the same hold set as a 12×12 TB2 Spray — just a different image and a smaller active area.
- Climbs set on a 12×12 are valid references on a 12×10 if all holds are within bounds.

Right now, to represent TB2 Mirror and TB2 Spray you need two separate walls with entirely duplicated hold sets — which is already the case in your live database. This also means climbs can't be shared or cross-referenced across sizes of the same layout.

### What boardlib teaches us

Studying the boardlib SQLite schemas (tension.db, kilter.db) reveals a well-considered hierarchy:

```
products             (brand-level: "Tension Board 2", "Kilter Board")
  └── layouts        (hold arrangement: "TB2 Mirror", "TB2 Spray", "Original Layout")
       └── placements (which holes exist in this layout, which set they belong to)
  └── product_sizes  (physical dimensions: "8×10", "12×12", "Full Wall")
       └── image_filename (per size)
  └── product_sizes_layouts_sets  (junction: size + layout + hold-set → image)

walls (boardlib)     (a user's specific physical board instance)
  → layout_id, product_size_id, hsm (bitmask of active sets)

climbs (boardlib)    → layout_id  (NOT size_id)
  + edge_left/right/bottom/top fields recording the size context
```

**Critical insight:** In boardlib, climbs belong to a *layout*, not a size. The size context is recorded on each climb, but climbs are discoverable across all sizes of a layout. Holds (placements) also belong to a layout and are the full master set — size only determines which holds are physically reachable on a given board.

---

## 2. Domain Model

### Proposed entity hierarchy for BetaZero

```
Layout  (the hold arrangement — what you currently call a "wall")
  id, name, description
  owner_id, visibility, share_token
  created_at, updated_at

  ├── Holds[]          (the full master set for this layout)
  │    hold_index, x, y, pull_x, pull_y, useability, is_foot, tags
  │
  ├── Sizes[]          (physical variants of this layout)
  │    id, layout_id
  │    name            ("8×12", "12×12", "Full Wall", "No Kickboard")
  │    width_ft, height_ft
  │    edge_left, edge_right, edge_bottom, edge_top  (in same units as hold coords)
  │    photo_path
  │    created_at, updated_at
  │
  └── Climbs[]         (tied to layout, optionally tagged with size context)
       id, layout_id, size_id (nullable)
       angle, name, holds (serialized), grade, quality, ...
```

### Entity definitions

**Layout** — a unique hold arrangement. Two boards with the same holes in the same positions are the same layout. The layout owns the holds. Multiple gyms/users can share a public layout (e.g., "Kilter Board Original 16×12").

**Size** — a physical variant of a layout. It has its own image (the board at that specific size) and edge bounds that define which holds from the layout's master set are actually present. A size does NOT have its own hold set; it filters the layout's holds by edge bounds.

**Hold** — belongs to the layout. Coordinates are in real-world units (feet). The full master hold set contains all holes that exist on the largest version of the layout. Smaller sizes simply exclude holds outside their edges.

**Climb** — belongs to the layout. An optional `size_id` records what size it was set on, which enables size-aware filtering (hide climbs that use holds out of bounds for my size), but climbs are still discoverable across all sizes by default.

### What happens to the current "wall"

The current BetaZero `walls` table maps to a layout **plus** its one size (since no existing wall has multiple sizes yet). The clean migration is:

- Each existing wall → one Layout + one Size (using the existing `dimensions` as the single size, existing `photo_path` as the size's photo).
- `holds.wall_id` → `holds.layout_id`
- `climbs.wall_id` → `climbs.layout_id`

No existing data is deleted. The migration is purely additive + rename.

---

## 3. Database Schema

### New tables

```sql
-- layouts: replaces walls (holds moved here, photo moved to sizes)
CREATE TABLE layouts (
    id           TEXT PRIMARY KEY,          -- "layout-{uuid12}"
    name         TEXT NOT NULL,
    description  TEXT,
    owner_id     TEXT NOT NULL,
    visibility   TEXT NOT NULL DEFAULT 'public',
    share_token  TEXT,
    num_holds    INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_id) REFERENCES users(id)
);

-- sizes: physical variants of a layout
CREATE TABLE sizes (
    id           TEXT PRIMARY KEY,          -- "size-{uuid12}"
    layout_id    TEXT NOT NULL,
    name         TEXT NOT NULL,             -- "8×12", "Full Wall", etc.
    width_ft     REAL,                      -- physical width in feet
    height_ft    REAL,                      -- physical height in feet
    edge_left    REAL NOT NULL DEFAULT 0.0, -- left bound in hold coord units
    edge_right   REAL,                      -- right bound (NULL = no clip)
    edge_bottom  REAL NOT NULL DEFAULT 0.0,
    edge_top     REAL,                      -- top bound (NULL = no clip)
    photo_path   TEXT,
    num_climbs   INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE
);
```

### Modified tables

```sql
-- holds: wall_id → layout_id (no other changes needed)
ALTER TABLE holds RENAME COLUMN wall_id TO layout_id;

-- climbs: wall_id → layout_id, add optional size context
ALTER TABLE climbs RENAME COLUMN wall_id TO layout_id;
ALTER TABLE climbs ADD COLUMN size_id TEXT REFERENCES sizes(id) ON DELETE SET NULL;
```

### Old walls table

After migration, the `walls` table can be dropped. During migration it is used as the source of truth to populate `layouts` + `sizes`.

### Storage layout on disk

```
data/
  walls/                         ← rename to layouts/ eventually
    layout-{id}/
      sizes/
        size-{id}/
          photo.jpg
      (no photo at layout root — photos live at size level)
```

For single-size layouts (all current walls), the directory is simply:

```
data/layouts/layout-{id}/sizes/size-{id}/photo.jpg
```

---

## 4. API Redesign

### Routing structure

```
/api/v1/layouts                          GET  list all layouts (+ their sizes)
                                         POST create layout (metadata only, no photo)

/api/v1/layouts/{layout_id}             GET  layout detail (metadata + holds + sizes list)
                                         DELETE layout + all sizes, holds, climbs

/api/v1/layouts/{layout_id}/holds       PUT  set/replace holds for layout
/api/v1/layouts/{layout_id}/climbs      GET  list climbs  (query: ?size_id=, ?angle=, ...)
                                         POST create climb
/api/v1/layouts/{layout_id}/generate    GET  generate climbs (query: ?size_id= for hold filter)

/api/v1/layouts/{layout_id}/sizes       GET  list sizes for layout
                                         POST create new size

/api/v1/layouts/{layout_id}/sizes/{size_id}          GET  size detail
                                                       PUT  update size metadata
                                                       DELETE size

/api/v1/layouts/{layout_id}/sizes/{size_id}/photo    GET  size photo
                                                       PUT  upload/replace size photo
```

### Key behavioral changes

**GET /layouts** returns each layout with a nested `sizes[]` list (id, name, dimensions, photo_url, num_climbs). This replaces GET /walls.

**GET /layouts/{layout_id}** returns the layout's full hold set (all holds, no size filtering) plus its sizes. Clients that need size-filtered holds compute the filter themselves using edge bounds, or pass `?size_id=` to get pre-filtered holds:

```
GET /layouts/{layout_id}?size_id={size_id}
```
returns only holds within that size's edge bounds.

**GET /layouts/{layout_id}/generate** accepts an optional `?size_id=` query param. When provided, the generation service filters the hold set to only those within the size's edge bounds before running the DDPM.

**GET /layouts/{layout_id}/climbs** accepts an optional `?size_id=` param for size-aware browsing. When `size_id` is provided, climbs that use holds outside the size's edges are excluded.

**POST /layouts/{layout_id}/sizes** (create size) accepts: `name`, `width_ft`, `height_ft`, `edge_left`, `edge_right`, `edge_bottom`, `edge_top`, and optionally a `photo` file.

### Backward compatibility

To avoid a hard cutover, add `/api/v1/walls` as a thin alias that proxies to `/api/v1/layouts` during the transition. This lets you migrate backend and frontend independently. Remove the alias once the frontend is fully updated.

---

## 5. Backend Code Changes

### New files

```
app/routers/layouts.py     ← replaces walls.py
app/routers/sizes.py       ← new
app/schemas/layouts.py     ← replaces schemas/walls.py
app/schemas/sizes.py       ← new
app/services/layout_service.py   ← replaces wall_service.py
app/services/size_service.py     ← new
```

### Modified files

```
app/database.py            ← add layouts + sizes tables, migration logic
app/main.py                ← re-register routers on new prefixes
app/services/generation_service.py  ← accept optional size_id, filter holds
app/services/climb_service.py       ← update wall_id → layout_id refs
app/schemas/climbs.py               ← add size_id field
```

### Generation service change (key)

The generation currently does:

```python
holds = get_holds(wall_id)  # all holds for this wall
```

After the change:

```python
holds = get_holds(layout_id)                    # full master hold set
if size_id:
    size = get_size(size_id)
    holds = filter_holds_by_size(holds, size)   # clip to edge bounds
```

`filter_holds_by_size` is a simple predicate:

```python
def filter_holds_by_size(holds, size):
    return [h for h in holds if
        (size.edge_left  is None or h.x >= size.edge_left)  and
        (size.edge_right is None or h.x <= size.edge_right) and
        (size.edge_bottom is None or h.y >= size.edge_bottom) and
        (size.edge_top    is None or h.y <= size.edge_top)]
```

---

## 6. Frontend Changes

### Type changes (`src/types/wall.ts`)

Rename and extend:

```typescript
// layouts.ts (new file, or extend wall.ts)

export interface SizeMetadata {
  id: string;
  layout_id: string;
  name: string;
  width_ft: number | null;
  height_ft: number | null;
  edge_left: number;
  edge_right: number | null;
  edge_bottom: number;
  edge_top: number | null;
  photo_url: string;
  num_climbs: number;
  created_at: string;
  updated_at: string;
}

export interface LayoutMetadata {
  id: string;
  name: string;
  description: string | null;
  num_holds: number;
  sizes: SizeMetadata[];   // always included in list response
  owner_id: string;
  visibility: string;
  share_token?: string;
  created_at: string;
  updated_at: string;
}

export interface LayoutDetail {
  metadata: LayoutMetadata;
  holds: HoldDetail[];     // full master set (or size-filtered if ?size_id provided)
}
```

### API client changes (`src/api/`)

```
src/api/walls.ts   → src/api/layouts.ts   (rename + update endpoints)
                     src/api/sizes.ts      (new: createSize, deleteSize, getSizePhotoUrl, uploadSizePhoto)
```

Key new function:

```typescript
export function getSizePhotoUrl(layoutId: string, sizeId: string): string {
  return `${BASE_URL}/layouts/${layoutId}/sizes/${sizeId}/photo`;
}
```

### Route changes (`src/routes/`)

Current routes:
```
/$wallId/set.tsx
/$wallId/view.tsx
/$wallId/holds.tsx
/walls/new.tsx
```

Proposed routes:
```
/$layoutId/
  sizes.tsx        ← NEW: size picker (shown when layout has >1 size)
  holds.tsx        ← unchanged in function, update to layout_id
  $sizeId/
    set.tsx        ← climb generator (needs both layoutId + sizeId)
    view.tsx       ← climb browser (needs both)
/layouts/new.tsx   ← rename from /walls/new
```

The `/$layoutId/$sizeId/set` nesting cleanly expresses that generation and browsing are size-scoped. The holds editor lives at the layout level because holds are shared.

**Single-size shortcut:** If a layout has exactly one size, navigating to `/$layoutId` redirects directly to `/$layoutId/$sizeId/set`, skipping the size picker. This keeps the UX identical to today for all current walls.

### Homepage changes (`src/routes/index.tsx`)

Currently renders a flat grid of wall cards. After the change:

- Cards represent **layouts** (not sizes).
- Each card shows the photo of the layout's *first/default size* (or a composite if desired).
- Sub-text shows size options: "3 sizes · 48 holds".
- Clicking navigates to `/$layoutId` (which redirects if single-size, shows picker if multi-size).

The "Add Your Wall" card navigates to `/layouts/new`.

### New wall creation wizard (`/walls/new.tsx` → `/layouts/new.tsx`)

Current steps: Upload → Crop → Details

Proposed steps: Details → Upload/Crop (first size) → Holds

The "Details" step now collects layout-level info (name, description, visibility) plus the first size's dimensions and name. After submit, the user is taken to the holds editor at `/$layoutId/holds`. Later, additional sizes can be added from a layout settings page.

### Hooks changes (`src/hooks/`)

```
useWalls.ts → useLayouts.ts   (rename, update types)
              useSizes.ts      (new: useSizes(layoutId) for size picker)
```

---

## 7. Migration Strategy

The migration must be non-destructive. All existing data survives. The steps:

### Step 1 — Database migration script (Python)

```python
# For each existing wall:
#   1. INSERT into layouts (same id, name, owner_id, visibility, share_token, num_holds, ...)
#   2. INSERT into sizes (new id, layout_id=wall.id, name="Default",
#                         width_ft and height_ft from wall.dimensions,
#                         edge_left=0, edge_right=width_ft, edge_bottom=0, edge_top=height_ft,
#                         photo_path=wall.photo_path)
#   3. Move photo file: walls/{wall_id}/photo.jpg → layouts/{wall_id}/sizes/{size_id}/photo.jpg
#   4. UPDATE holds SET layout_id = wall_id WHERE wall_id = wall.id
#   5. UPDATE climbs SET layout_id = wall_id WHERE wall_id = wall.id
#   6. DROP TABLE walls (after verifying all data migrated)
```

Critically: since existing wall IDs become layout IDs, all existing URLs remain valid. No broken links.

### Step 2 — Backend: deploy new schema alongside old

Add `layouts` and `sizes` tables. Keep `walls` table. Run migration. Serve both old `/walls/` and new `/layouts/` endpoints simultaneously (aliases). Verify.

### Step 3 — Frontend: update to new endpoints and routes

Update types, API client, hooks, and routes. Deploy. Verify all existing wall IDs still resolve (via the redirect logic — `/$id` checks if it's a layout or a size and routes accordingly).

### Step 4 — Cleanup

Remove `/walls/` alias endpoints. Drop `walls` table. Done.

---

## 8. Open Questions & Decisions

These need a product decision before implementation:

**Q1: Should "layout" be the word used in URLs and UI, or keep "wall"?**
The user expressed willingness to call layouts "walls." Option: keep `wall` as the URL segment (backward compatible) and use "layout" only internally. The homepage would still say "Choose your wall" and the URL stays `/$wallId/...`. Sizes get a new segment: `/$wallId/$sizeId/set`.

This is the lowest-friction path. Recommended unless you want the cleaner conceptual split.

**Q2: Should climbs be tied to `layout_id` or `size_id`?**
Boardlib ties climbs to `layout_id`, which allows cross-size discovery. This is the right call for BetaZero too — a V6 set on a 12×12 is still a V6, even if you're on an 8×12 (with the caveat that some holds may be out of bounds). Recommended: `layout_id` with optional `size_id` as context.

**Q3: For the generation model, should size filtering happen at the API or in the model?**
At the API. The DDPM model itself is agnostic — it just gets a list of holds and generates. The API pre-filters the hold list based on size edges before calling generate. This is architecturally clean and requires zero model changes.

**Q4: Who can add sizes to a layout?**
The layout owner only (consistent with current wall ownership). A separate concept of "organization" ownership could come later if needed.

**Q5: What about user-created layouts on commercial boards?**
BetaZero currently allows any user to create a wall. Under the new model, a user could create a "Kilter Board 12×12" layout that duplicates an existing one. This is acceptable for now — deduplication/linking can be a future feature (e.g., "is this a known commercial board?").

---

## 9. Development Roadmap

### Phase 1 — Database & Migration (backend only, ~1 session)

1. Add `layouts` and `sizes` tables to `database.py` → `init_db()`.
2. Write `db_migration.py` script to migrate existing walls → layouts + sizes.
3. Run migration against dev DB. Verify data integrity.
4. Update `holds` table: `wall_id` → `layout_id` (SQLite rename via migration).
5. Update `climbs` table: `wall_id` → `layout_id`, add `size_id` column.

**Deliverable:** DB migrated, all old data intact, old `walls` table dropped.

---

### Phase 2 — Backend: new schemas and services (~1 session)

1. Create `app/schemas/layouts.py` — `LayoutMetadata`, `LayoutCreate`, `LayoutDetail`, `LayoutListResponse`.
2. Create `app/schemas/sizes.py` — `SizeMetadata`, `SizeCreate`, `SizeDetail`.
3. Create `app/services/layout_service.py` — port `wall_service.py`, replace `wall_id` with `layout_id`, add size-aware `get_holds(layout_id, size_id=None)`.
4. Create `app/services/size_service.py` — CRUD for sizes, photo management.
5. Update `app/services/generation_service.py` — accept `size_id`, apply edge-bound filtering.
6. Update `app/services/climb_service.py` — `wall_id` → `layout_id`, add `size_id` filter in `get_climbs`.

**Deliverable:** All service logic updated, old `wall_service.py` preserved but deprecated.

---

### Phase 3 — Backend: new API routes (~1 session)

1. Create `app/routers/layouts.py` — all `/layouts/...` endpoints.
2. Create `app/routers/sizes.py` — all `/layouts/{layout_id}/sizes/...` endpoints.
3. Update `app/main.py` — register new routers, keep old `/walls/` as alias redirects temporarily.
4. Update `app/test/test_api.py` — update tests to use new routes.
5. Manual smoke test all endpoints.

**Deliverable:** Full API functional on new routes. Old routes still respond (proxied).

---

### Phase 4 — Frontend: types and API client (~1 session)

1. Add `src/types/layout.ts` — `LayoutMetadata`, `LayoutDetail`, `SizeMetadata`, `LayoutListResponse`, etc.
2. Update `src/types/wall.ts` — keep for backward compat during transition, mark deprecated.
3. Add `src/api/layouts.ts` — `getLayouts()`, `getLayout(id)`, `createLayout()`, `deleteLayout()`.
4. Add `src/api/sizes.ts` — `createSize()`, `deleteSize()`, `getSizePhotoUrl()`, `uploadSizePhoto()`.
5. Update `src/api/walls.ts` — proxy to layouts.ts during transition.
6. Run `npx tsc --noEmit` — fix all type errors.

**Deliverable:** Frontend compiles with new types and API client.

---

### Phase 5 — Frontend: routes and pages (~2 sessions)

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

### Phase 6 — Cleanup (~0.5 session)

1. Remove `/api/v1/walls/` alias endpoints from backend.
2. Remove deprecated `src/types/wall.ts` (or reduce to a re-export of layout types).
3. Remove deprecated `src/api/walls.ts` (or reduce to re-exports).
4. Remove deprecated `wall_service.py` and `routers/walls.py`.
5. Final `npx tsc --noEmit`. Final backend test run.
6. Commit and tag.

**Deliverable:** Clean codebase with no dead code, full layouts+sizes support.

---

## 10. Summary

| Concern | Current | After |
|---|---|---|
| Primary entity | `wall` (conflates layout + size) | `layout` (hold arrangement) |
| Size info | `dimensions` on wall | `sizes` table, child of layout |
| Photo | one per wall | one per size |
| Holds | per wall_id | per layout_id (shared across sizes) |
| Climbs | per wall_id | per layout_id, optional size_id |
| Generation | uses all holds of wall | uses holds filtered by size edges |
| Homepage | flat wall grid | layout grid (size count shown per card) |
| URL: generator | `/$wallId/set` | `/$layoutId/$sizeId/set` |
| URL: holds editor | `/$wallId/holds` | `/$layoutId/holds` |
| Migration cost | — | additive; no data loss; old IDs reused as layout IDs |

The migration is low-risk. All existing wall IDs become layout IDs. Every existing "wall" becomes a layout with exactly one size. The front-end and back-end can be migrated independently behind the alias layer. No user-visible breakage is expected.
