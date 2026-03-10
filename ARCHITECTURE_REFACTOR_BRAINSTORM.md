# BetaZero — Architecture Refactor Brainstorm

> A candid architectural review of the current codebase with concrete improvement proposals.
> Written as a planning document — nothing here is a committed plan.

---

## Table of Contents

1. [Frontend](#1-frontend)
   - 1.1 [GLOBAL_STYLES: Consolidate to a single injection point](#11-global_styles-consolidate-to-a-single-injection-point)
   - 1.2 [Monolithic route files: break them up](#12-monolithic-route-files-break-them-up)
   - 1.3 [Coordinate conversion: extract to a shared utility](#13-coordinate-conversion-extract-to-a-shared-utility)
   - 1.4 [Typed tuples for dimensions and edges](#14-typed-tuples-for-dimensions-and-edges)
   - 1.5 [Canvas drawing: extract a CanvasPainter utility](#15-canvas-drawing-extract-a-canvaspainter-utility)
   - 1.6 [fetchWithWakeRetry belongs in the API layer](#16-fetchwithwakeretry-belongs-in-the-api-layer)
   - 1.7 [Inline styling: commit to a strategy](#17-inline-styling-commit-to-a-strategy)
   - 1.8 [Finish the Wall → Layout naming migration](#18-finish-the-wall--layout-naming-migration)
2. [Backend](#2-backend)
   - 2.1 [Auth enforcement is incomplete — fix it](#21-auth-enforcement-is-incomplete--fix-it)
   - 2.2 [Form + JSON.loads() → Pydantic JSON bodies](#22-form--jsonloads--pydantic-json-bodies)
   - 2.3 [Dynamic SQL building → a simple QueryBuilder](#23-dynamic-sql-building--a-simple-querybuilder)
   - 2.4 [Ad-hoc schema migrations → Alembic](#24-ad-hoc-schema-migrations--alembic)
   - 2.5 [Database schema hardening](#25-database-schema-hardening)
3. [Cross-cutting concerns](#3-cross-cutting-concerns)
   - 3.1 [Structured error codes end-to-end](#31-structured-error-codes-end-to-end)
   - 3.2 [Document the coordinate system once, formally](#32-document-the-coordinate-system-once-formally)

---

## 1. Frontend

### 1.1 GLOBAL_STYLES: Consolidate to a single injection point

**Problem**

`GLOBAL_STYLES` (the CSS variable block that defines the entire BetaZero design token set) is currently defined in three separate locations:

- `src/components/wall/styles.ts`
- `src/styles.ts` (or similar)
- Inline in `src/routes/layouts/new.tsx`

Every page that needs the design tokens does `<style>{GLOBAL_STYLES}</style>`, which means the browser is parsing and injecting the same ~120-line CSS string multiple times per render cycle — once per mounted route. This also creates a maintenance hazard where a change to one copy doesn't propagate to the others.

**Proposal**

Inject `GLOBAL_STYLES` exactly once, in `__root.tsx`, at the very root of the component tree. Delete all other definitions and all per-page `<style>` injections.

```tsx
// __root.tsx
import { GLOBAL_STYLES } from "@/styles";

export function RootComponent() {
  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <Outlet />
    </>
  );
}
```

Every route then simply uses `var(--cyan)` etc. in its inline styles or CSS classes — no import required. This is a ~30-minute change with zero risk and immediate payoff in both correctness and bundle cleanliness.

---

### 1.2 Monolithic route files: break them up

**Problem**

Several route files have grown into large, multi-concern monoliths:

| File | Lines | Distinct concerns |
|------|-------|-------------------|
| `$layoutId/set.tsx` | ~2,663 | Generation settings, display settings, climb navigation, mobile swipe, share/export |
| `$layoutId/holds.tsx` | ~979 | Canvas interaction, hold state, mode toolbar, sidebar, tag system |
| `layouts/new.tsx` | ~1,000 | 4-step wizard state, each step's UI, photo upload, cropping |

These aren't just cosmetically large — they're hard to navigate, hard to test, and create merge conflicts when two features are touched simultaneously.

**Proposal: `set.tsx`**

Extract three feature-level components and two hooks:

```
src/
  routes/$layoutId/set.tsx          ← thin orchestrator only (~150 lines)
  features/generator/
    GeneratorPanel.tsx              ← all generation settings UI
    ClimbNavigator.tsx              ← previous/next, swipe gestures
    ExportMenu.tsx                  ← image export logic
    useClimbGeneration.ts           ← generation request state
    useClimbNavigation.ts           ← climb list, current index
```

`set.tsx` becomes a composition root that wires these together — it holds no business logic itself.

**Proposal: `new.tsx`**

The 4-step wizard is a natural split:

```
src/
  routes/layouts/new.tsx            ← wizard shell (step state, navigation)
  features/new-layout/
    VisibilityStep.tsx
    UploadStep.tsx
    CropStep.tsx
    DetailsStep.tsx
    useLayoutCreationWizard.ts      ← step state, validation, submit logic
```

**Proposal: `holds.tsx`**

The holds editor canvas logic and the toolbar/sidebar UI are already partially separated (sidebar lives in `components/holds.tsx`), but the route file itself still manages too much. The canvas event handlers and drawing loop could move into a dedicated hook:

```
src/
  hooks/useHoldCanvas.ts            ← canvas draw loop, mouse event handlers
  routes/$layoutId/holds.tsx        ← wires useHolds + useHoldCanvas + sidebar
```

---

### 1.3 Coordinate conversion: extract to a shared utility

**Problem**

The pixel↔feet coordinate conversion (`toPixelCoords` / `toFeetCoords`) exists in at least two places:

- `src/hooks/useHolds.ts` — computes feet coords from pixel input
- `src/components/wall/WallCanvas.tsx` — recomputes pixel coords from feet for rendering

If the wall coordinate system ever changes (e.g. a different edge-crop calculation or a new origin convention), both must be updated in sync — and it's not obvious they're coupled.

**Proposal**

Create a pure utility module with no React dependencies:

```ts
// src/utils/coordinates.ts

export type FeetPoint = { x: number; y: number };
export type PixelPoint = { x: number; y: number };

export interface CoordContext {
  imageDims: { width: number; height: number };
  wallDims: { width: number; height: number };
  edges: [left: number, right: number, bottom: number, top: number] | null;
}

export function toFeet(pixel: PixelPoint, ctx: CoordContext): FeetPoint { ... }
export function toPixels(feet: FeetPoint, ctx: CoordContext): PixelPoint { ... }
```

Both `useHolds` and `WallCanvas` import from here. The single source of truth for the conversion lives in one well-documented place.

---

### 1.4 Typed tuples for dimensions and edges

**Problem**

Two critical data shapes are typed as plain arrays:

- `dimensions: number[]` — represents `[width_ft, height_ft]`
- `edges: number[]` — represents `[left, right, bottom, top]` in feet

At array access sites (`dims[0]`, `edges[2]`), there is no compiler help if an index is wrong. A typo like `edges[1]` instead of `edges[2]` for the bottom edge produces no warning.

**Proposal**

Define named tuple types in `src/types/layout.ts`:

```ts
export type WallDimensions = [widthFt: number, heightFt: number];
export type EdgeCrop = [leftFt: number, rightFt: number, bottomFt: number, topFt: number];
```

Update `LayoutMetadata.dimensions` and `SizeMetadata.edges` to use these types. TypeScript will now catch most index errors at compile time, and the named labels appear in IDE hover tooltips.

This is a mechanical find-and-replace plus type annotation update — moderate effort, high long-term value.

---

### 1.5 Canvas drawing: extract a CanvasPainter utility

**Problem**

Both `WallCanvas.tsx` and `$layoutId/holds.tsx` contain large `useEffect` blocks that directly call Canvas 2D API methods (`ctx.beginPath()`, `ctx.arc()`, `ctx.strokeStyle = ...`, etc.). This makes the drawing logic:

- Hard to test (requires a real or mocked canvas)
- Scattered across two large files
- Difficult to reuse (e.g. drawing a hold circle the same way in both places)

**Proposal**

Extract a set of pure drawing functions into `src/utils/canvasDraw.ts`:

```ts
export function drawHoldCircle(ctx: CanvasRenderingContext2D, x: number, y: number, opts: HoldDrawOpts): void { ... }
export function drawPullArrow(ctx: CanvasRenderingContext2D, hold: HoldDetail, opts: ArrowDrawOpts): void { ... }
export function drawSelectionRing(ctx: CanvasRenderingContext2D, x: number, y: number, size: number): void { ... }
export function drawTagIndicator(ctx: CanvasRenderingContext2D, x: number, y: number, size: number): void { ... }
```

These are plain functions (no hooks, no React) — fully unit-testable with a mock canvas context. The `useEffect` drawing loops in both files become ~10 lines of function calls instead of 100+ lines of raw canvas operations. Consistent visual output is also much easier to guarantee.

---

### 1.6 `fetchWithWakeRetry` belongs in the API layer

**Problem**

`fetchWithWakeRetry` — the logic that retries a fetch when the backend returns 502 (cold start wake-up) — currently lives inside `src/hooks/useLayouts.ts`. But it's not specific to layouts: any API call could need this behavior (fetching climbs, generating a route, etc.).

**Proposal**

Move it into `src/api/client.ts`, where it belongs:

```ts
// src/api/client.ts
export async function fetchWithWakeRetry<T>(
  fn: () => Promise<T>,
  onWaking?: () => void,
): Promise<T> { ... }
```

Every hook (`useLayouts`, `useClimbs`, etc.) can then import and use it. The retry logic is defined once and applied consistently across the entire API layer. `useLayouts.ts` shrinks and loses a responsibility it shouldn't own.

---

### 1.7 Inline styling: commit to a strategy

**Problem**

The codebase uses inline `style={{}}` with CSS variable strings almost everywhere in the frontend. This is functional, but creates a few friction points:

- No autocomplete for CSS property names in string templates
- No static analysis of unused styles
- Responsive/hover/focus states require `onMouseEnter`/`onMouseLeave` handlers (see the many hover handlers throughout `holds.tsx`)
- Visual consistency depends on discipline, not tooling

The project already has Tailwind installed but it's used minimally.

**Proposal options (pick one direction and commit):**

**Option A — Lean into Tailwind fully.** Tailwind already handles the hover/focus problem cleanly. The BetaZero design tokens would live in `tailwind.config.ts` as custom color/spacing values (`bg-bz-surface`, `text-bz-cyan`, etc.). Components use class strings instead of `style={}`. This is the highest-effort but most sustainable option long-term.

**Option B — CSS Modules per component.** Each component gets a `.module.css` file. Design tokens stay as CSS variables (still injected from `__root.tsx`). This eliminates hover handler noise and gives you scoped class names without the Tailwind learning curve.

**Option C — Keep inline styles, add a style helper.** At minimum, factor out the repeated button/card style patterns into typed helper functions (like the existing `modeButtonStyle` in `holds.tsx`), so changes are made in one place. Low effort, incremental improvement.

The current situation — some Tailwind, mostly inline, some CSS variable strings — is the worst of all options because nothing is consistent.

---

### 1.8 Finish the Wall → Layout naming migration

**Problem**

The Wall → Layout renaming is mostly complete but has some stragglers:

- `src/types/wall.ts` still exports `WallMetadata`, `WallDetail`
- `src/types/climb.ts` has `wall_id` on the `Climb` type (should be `layout_id`)
- `src/api/climbs.ts` may still call `/walls/{id}/climbs` endpoints
- `climb-backend/app/auth.py` has `get_accessible_wall` alongside `get_accessible_layout`

These dual-naming remnants make it harder to search for anything — grep results mix old and new names.

**Proposal**

Audit all `wall` references (case-insensitive) in `src/` and `app/`. For each one, determine whether it's:
- A legitimate legacy route alias (keep, but mark with `# legacy` comment)
- A stale type name (rename)
- A dead code path (delete)

The goal is that searching for "wall" in the codebase only finds intentional backward-compatibility aliases, not accidental omissions.

---

## 2. Backend

### 2.1 Auth enforcement is incomplete — fix it

**Problem**

In `routers/layouts.py` (and likely `routers/climbs.py`), multiple endpoint handlers have auth dependencies that were commented out:

```python
# _=Depends(require_auth)   ← commented out
```

This means unauthenticated users can currently delete layouts, modify holds, and create climbs. The `auth.py` module has correct logic already written — it's just not being applied.

**Proposal**

Re-enable the auth dependencies on all mutating endpoints (POST, PUT, DELETE). Read endpoints can remain public if that's the intended design.

At the router level, FastAPI's `dependencies` argument at the router-init level can apply auth to all routes in a router at once, rather than per-endpoint:

```python
router = APIRouter(
    prefix="/layouts",
    dependencies=[Depends(require_auth)],  # applies to all routes
)
```

Individual endpoints that should be public (e.g. GET /layouts) can override with an explicit `dependencies=[]`.

This is probably the single highest-impact change in the whole codebase from a correctness standpoint.

---

### 2.2 Form + `JSON.loads()` → Pydantic JSON bodies

**Problem**

Several endpoints accept JSON data through HTML Form fields with manual parsing:

```python
# routers/layouts.py
async def create_layout(
    name: str = Form(...),
    dimensions: str = Form(...),   # ← JSON string
    image_edges: str = Form(...),  # ← JSON string
):
    dims = json.loads(dimensions)
    edges = json.loads(image_edges)
```

This is the pattern that exists because photo uploads need `multipart/form-data`. But for endpoints that don't upload a file, there's no reason to use Form fields at all — they should accept a JSON body with a Pydantic model.

**Proposal**

Split endpoints that mix file upload with structured data into two requests where appropriate, or use a mixed approach:

```python
# For non-file endpoints: clean Pydantic model
class LayoutCreateRequest(BaseModel):
    name: str
    dimensions: tuple[float, float]
    visibility: str = "private"

@router.post("/")
async def create_layout(body: LayoutCreateRequest, user=Depends(require_auth)):
    ...
```

For the photo upload endpoint (which legitimately needs multipart), keep Form fields but add clear validation:

```python
@router.put("/{id}/photo")
async def upload_photo(
    id: str,
    photo: UploadFile = File(...),
    user=Depends(require_auth),
):
    ...
```

This makes the API surface cleaner and removes several `json.loads()` calls and their associated error cases.

---

### 2.3 Dynamic SQL building → a simple QueryBuilder

**Problem**

`services/climb_service.py` and `services/layout_service.py` both contain long blocks of conditional SQL string construction:

```python
where_clauses = []
params = []

if min_grade is not None:
    where_clauses.append("grade_value >= ?")
    params.append(min_grade)
if max_grade is not None:
    where_clauses.append("grade_value <= ?")
    params.append(max_grade)
if setter_name:
    where_clauses.append("setter_name = ?")
    params.append(setter_name)
# ... 10 more of these

query = "SELECT * FROM climbs"
if where_clauses:
    query += " WHERE " + " AND ".join(where_clauses)
```

This pattern is repeated in multiple services and is a known footgun — the clause list and params list must stay in sync manually, which is error-prone and hard to test.

**Proposal**

A minimal `QueryBuilder` class (no third-party dependency needed):

```python
# app/db/query_builder.py

class QueryBuilder:
    def __init__(self, table: str):
        self._table = table
        self._conditions: list[str] = []
        self._params: list = []
        self._order: str | None = None
        self._limit: int | None = None

    def where(self, clause: str, *params) -> "QueryBuilder":
        self._conditions.append(clause)
        self._params.extend(params)
        return self

    def where_if(self, condition, clause: str, *params) -> "QueryBuilder":
        if condition:
            self.where(clause, *params)
        return self

    def order_by(self, col: str, desc: bool = False) -> "QueryBuilder":
        self._order = f"{col} {'DESC' if desc else 'ASC'}"
        return self

    def build_select(self, cols="*") -> tuple[str, list]:
        sql = f"SELECT {cols} FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        if self._order:
            sql += f" ORDER BY {self._order}"
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql, self._params
```

Usage becomes:

```python
sql, params = (
    QueryBuilder("climbs")
    .where("layout_id = ?", layout_id)
    .where_if(min_grade, "grade_value >= ?", min_grade)
    .where_if(max_grade, "grade_value <= ?", max_grade)
    .where_if(setter_name, "setter_name = ?", setter_name)
    .order_by("created_at", desc=True)
    .build_select()
)
```

This is about 50 lines of infrastructure code that makes every filtering function in the services significantly shorter and less prone to parameter/clause desync bugs.

---

### 2.4 Ad-hoc schema migrations → Alembic

**Problem**

`database.py` contains a custom migration mechanism:

```python
def _column_exists(conn, table, column) -> bool: ...
def _add_column_if_missing(conn, table, column, definition): ...
```

These are called on startup to add new columns as the schema evolves. This works fine for simple additive changes, but breaks down for:
- Column renames
- Column type changes
- Adding constraints to existing columns
- Adding indexes
- Dropping columns

More importantly, there's no migration history — no way to know what state a given database is in relative to the current schema, or to roll back a bad migration.

**Proposal**

Migrate to [Alembic](https://alembic.sqlalchemy.org/), which integrates cleanly with SQLite and provides:
- Sequential, versioned migration files
- Upgrade and downgrade paths
- A revision graph that makes schema history auditable

The initial migration can be generated from the current `init_db()` schema, and all future `_add_column_if_missing` calls become proper Alembic migration scripts. This is a moderate-effort change that pays dividends immediately every time the schema needs to evolve.

---

### 2.5 Database schema hardening

**Problem**

The current SQLite schema has several soft spots worth addressing:

**No foreign key constraints defined.** SQLite has FK support but requires `PRAGMA foreign_keys = ON` AND explicit `FOREIGN KEY (col) REFERENCES table(id)` clauses in the `CREATE TABLE` statements. If a layout is deleted but holds rows aren't, the database silently becomes inconsistent.

**Dimensions stored as INTEGER.** The schema stores `dimensions` as an INTEGER column but the actual data is a JSON string (or possibly a serialized list). This is a type mismatch that silently accepts wrong data types.

**No indexes on hot query paths.** `climbs` is queried by `layout_id`, `owner_id`, and `created_at` constantly. Without indexes, every filter is a full table scan — fine now with small data, but will degrade noticeably at a few thousand climbs.

**Proposal**

```sql
-- Explicit FK constraints
CREATE TABLE holds (
    ...
    layout_id TEXT NOT NULL REFERENCES layouts(id) ON DELETE CASCADE,
    ...
);

CREATE TABLE climbs (
    ...
    layout_id TEXT NOT NULL REFERENCES layouts(id) ON DELETE CASCADE,
    size_id   TEXT REFERENCES sizes(id) ON DELETE SET NULL,
    ...
);

-- Correct type for dimensions
ALTER TABLE layouts ADD COLUMN width_ft REAL;
ALTER TABLE layouts ADD COLUMN height_ft REAL;
-- (migrate data from JSON string, then drop old column)

-- Indexes
CREATE INDEX IF NOT EXISTS idx_climbs_layout_id ON climbs(layout_id);
CREATE INDEX IF NOT EXISTS idx_climbs_owner_id  ON climbs(owner_id);
CREATE INDEX IF NOT EXISTS idx_climbs_created_at ON climbs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_holds_layout_id  ON holds(layout_id);
```

These changes are mostly additive and low-risk, but would be best applied through Alembic (see §2.4) rather than manually.

---

## 3. Cross-cutting concerns

### 3.1 Structured error codes end-to-end

**Problem**

Errors are currently communicated as freeform strings:

```python
# backend
raise HTTPException(status_code=404, detail="Layout not found")
```

```ts
// frontend
setError(err instanceof Error ? err.message : "Failed to save holds");
```

This creates two problems:
1. **Presentation coupling**: The UI displays raw backend error strings directly, meaning backend phrasing changes break UX consistency.
2. **Actionability**: The frontend can't programmatically distinguish "layout not found" from "unauthorized" from "photo too large" without string matching.

**Proposal**

Define a small set of error codes on the backend:

```python
# app/errors.py
class BetaZeroError(HTTPException):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail={"code": code, "message": message})

LAYOUT_NOT_FOUND = lambda: BetaZeroError("LAYOUT_NOT_FOUND", "Layout not found", 404)
UNAUTHORIZED     = lambda: BetaZeroError("UNAUTHORIZED", "Authentication required", 401)
PHOTO_TOO_LARGE  = lambda: BetaZeroError("PHOTO_TOO_LARGE", "Photo must be under 10MB", 413)
```

On the frontend, the API client reads `error.response.data.code` and maps it to a user-facing message string. This decouples backend wording from frontend display, and allows the frontend to react differently to different error types (e.g. redirect to login on UNAUTHORIZED, show inline warning on PHOTO_TOO_LARGE).

---

### 3.2 Document the coordinate system once, formally

**Problem**

BetaZero uses two coordinate systems simultaneously:

- **Pixel space**: origin top-left, y increases downward — native to HTML canvas
- **Wall space (feet)**: origin bottom-left, y increases upward — what's stored in the database and shown to users

The conversion between them depends on `imageDimensions`, `wallDimensions`, and optionally `imageEdges` (for size crop offsets). This math is duplicated across `useHolds.ts`, `WallCanvas.tsx`, and potentially `sizes.tsx`.

Developers working on any canvas-touching feature must understand both systems and the conversion — but this is currently undocumented and learned by reading code.

**Proposal**

Add a comment block to the top of `src/utils/coordinates.ts` (see §1.3) that formally defines both coordinate systems, the conversion formulas, and the edge-crop offset:

```ts
/**
 * BETAZERO COORDINATE SYSTEMS
 *
 * PIXEL SPACE
 *   Origin: top-left corner of the wall photo
 *   +x: right, +y: down
 *   Units: pixels
 *
 * WALL SPACE
 *   Origin: bottom-left corner of the physical wall
 *   +x: right (toward right edge), +y: up (toward top of wall)
 *   Units: feet
 *   Stored in: holds.x, holds.y, size.edges, layout.dimensions
 *
 * EDGE CROP OFFSET
 *   When a size has edges=[left, right, bottom, top], the photo is cropped
 *   such that pixel (0,0) corresponds to wall point (left, bottom).
 *   The visible wall width = right - left, height = top - bottom.
 *
 * CONVERSION (pixel → feet)
 *   x_ft = left_ft + (px / imgWidth)  * (right_ft - left_ft)
 *   y_ft = bottom_ft + (1 - py / imgHeight) * (top_ft - bottom_ft)
 */
```

This is zero-code-change documentation that will save every future contributor 20+ minutes of orientation time.

---

## Priority Summary

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 2.1 | Re-enable auth enforcement | Low | Critical |
| 1.1 | Consolidate GLOBAL_STYLES | Low | High |
| 1.3 | Extract coordinate utility | Low | High |
| 3.2 | Document coordinate system | Low | Medium |
| 1.8 | Finish Wall → Layout rename | Low | Medium |
| 1.4 | Typed tuples for dims/edges | Low–Med | Medium |
| 1.6 | Move fetchWithWakeRetry to API layer | Low | Medium |
| 2.2 | Form → Pydantic JSON bodies | Medium | High |
| 2.3 | QueryBuilder for dynamic SQL | Medium | Medium |
| 2.5 | Database schema hardening | Medium | High |
| 1.2 | Break up monolithic routes | High | High |
| 1.5 | Canvas drawing utility | Medium | Medium |
| 3.1 | Structured error codes | Medium | Medium |
| 2.4 | Alembic migrations | Medium | Medium |
| 1.7 | Commit to a styling strategy | High | Medium |
