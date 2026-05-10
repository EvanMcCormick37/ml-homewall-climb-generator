# BetaZero — Project Context

BetaZero is a full-stack web app that uses a DDPM (diffusion model) to generate climbing routes on homewalls and system boards. Users upload a photo of their wall, place holds on a canvas, then generate or manually set climbs and save them to a shared database.

Live at: https://betazero.live

---

## Product Domain

**Entities:**
- **Layout** — a physical wall: photo, dimensions (ft), hold positions, visibility (`public` / `unlisted` / `private`)
- **Size** — a sub-region of a layout (e.g. with/without kickboard); defined by edge bounds in feet
- **Hold** — a single hold on a layout: position in feet (origin: bottom-left), pull direction, useability rating, role (`hand` / `foot`), tags
- **Climb** — a route on a layout: a `Holdset` (start/finish/hand/foot hold indices), grade, angle, setter, quality, tags
- **Holdset** — `{ start: number[], finish: number[], hand: number[], foot: number[] }` — indexes into the layout's hold array

**Grade system:** grades stored as floats via `GRADE_TO_DIFF` map. Supported scales: `v_grade` (V0–V17), `font` (1a–9a+).

**Coordinate system:** holds stored in **feet** (origin: bottom-left). Canvas renders in **pixels** (origin: top-left). Y is inverted between the two systems.

---

## Architecture

Monorepo with two packages:

```
ml-homewall-climb-generator/
├── climb-backend/     # FastAPI Python backend
└── climb-frontend/    # React + TypeScript frontend
```

---

## Backend (`climb-backend`)

**Stack:** FastAPI 0.125 · Uvicorn · SQLite 3 (raw SQL) · PyJWT / Clerk · PyTorch 2.10 (CPU) · scikit-learn · pydantic-settings · httpx

**Entry point:** `app/main.py` — registers CORS (all origins), mounts all routers under `/api/v1`, calls `init_db()` on startup.

**Layer breakdown:**
- `routers/` — thin HTTP handlers; no raw SQL; delegate to services via `ServiceContainer`
- `services/` — all business logic; injected via FastAPI DI
- `schemas/` — Pydantic request/response models
- `app/database.py` — schema init + migration helpers
- `app/auth.py` — Clerk JWT verification (RS256, JWKS cached 1 h); FastAPI dependency functions

**Key services:**
| Service | Responsibility |
|---|---|
| `layout_service` | Layout + hold CRUD; photo storage at `data/layouts/{id}/` |
| `climb_service` | Climb CRUD with rich filtering; batch insert |
| `generation_service` | Acquires generator from pool; runs DDPM; converts output to Holdset |
| `user_service` | `ensure_user_exists` upsert on every authenticated request |
| `size_service` | Size CRUD |

**Database:** SQLite at `data/storage.db`. JSON arrays (holds, edges, tags) stored as serialized strings. Tables: `users`, `layouts`, `sizes`, `holds`, `climbs`.

**File storage:**
```
data/
├── storage.db
├── layouts/{layout_id}/photo.jpg + photo-small.jpg
└── models/ddpm-weights.pth + scaler-weights.joblib + unet-hold-classifier.pth
```

**Auth dependency chain:**
- `get_current_user` → extracts + verifies Bearer token (returns `None` if absent)
- `require_auth` → 401 if unauthenticated
- `sync_auth` → `require_auth` + upserts user into DB
- `get_accessible_layout` → 404/403 based on visibility + ownership
- `require_layout_owner` → 401/403

**Layout access rules:** `public` = anyone; `unlisted` = anyone with `?share_token=`; `private` = owner or valid share token.

**ML generation pipeline:**
1. Load hold positions from DB
2. DDPM conditions on `grade` + `angle`, denoises noise → point cloud
3. UNet hold classifier assigns roles (start / finish / hand / foot)
4. Manifold guidance snaps points to nearest real hold positions
5. Returns `Holdset` (hold indices)

Thread-safe `Generator` pool (default size: 4). Models lazy-loaded on first request. Generation runs synchronously (~2–5 s). Async job queue is a planned future feature.

**API prefix:** `/api/v1`

Key route groups:
- `GET/POST/PUT/DELETE /layouts` + photo/holds sub-routes
- `GET/POST/DELETE /layouts/{id}/sizes`
- `GET/POST/DELETE /layouts/{id}/climbs` (+ `/batch`)
- `GET /layouts/{id}/generate` — query params: `num_climbs`, `grade`, `grade_scale`, `angle`, `timesteps`, `t_start_projection`, `x_offset`, `guidance_value`, `deterministic`, `seed`
- `GET /health`

**Config (`.env` via pydantic-settings):** `CLERK_ISSUER`, `CLERK_SECRET_KEY`, `DATA_DIR`, `LAYOUTS_DIR`, `DB_PATH`, `DDPM_WEIGHTS_PATH`, `SCALER_WEIGHTS_PATH`, `HC_WEIGHTS_PATH`, `GENERATOR_POOL_SIZE` (4), `LIMIT` (50).

---

## Frontend (`climb-frontend`)

**Stack:** React 19 · TypeScript 5.9 · Vite 7.2 · TanStack Router 1.143 (file-based) · Clerk 5.61 · Axios 1.13 · Tailwind CSS 4

**Routes:**
| Route | Page |
|---|---|
| `/` | Homepage — layout grid, hero, "Add Your Wall" |
| `/layouts/new` | 4-step creation wizard (visibility → photo → crop → details) |
| `/$layoutId/holds` | Interactive hold placement canvas |
| `/$layoutId/set` | Climb generator (grade/angle/settings → generate → save) |
| `/$layoutId/view` | Saved climb browser (filters, sort, canvas visualization) |
| `/$layoutId/sizes` | Size variant manager |

**API client:** `src/api/client.ts` — Axios instance pointed at `VITE_API_URL`. Request interceptor injects `Authorization: Bearer {token}`. Token provider wired in from Clerk via `setAuthTokenProvider()` in `__root.tsx`.

**Key hooks:**
| Hook | Purpose |
|---|---|
| `useBetaZeroAuth` | Clerk auth state + `getApiToken` |
| `useLayouts` / `useLayout` | Fetch layouts; includes 502 wake-retry for cold-start hosting |
| `useClimbs` | Climb list with filters, delete, selected climb state |
| `useHolds` | Hold edit state + pixel↔feet coordinate conversion |
| `useImageCrop` | Canvas-based crop drag state |

**Key components:**
- `WallCanvas` — renders layout photo + hold markers + climb overlay
- `ImageCropper` / `TrapezoidCropper` — crop/perspective tools in the creation wizard
- `DisplaySettingsPanel` — hold/climb display settings
- `SaveShareMenu` — save climb + copy share link
- `WakingScreen` — spinner overlay for server cold-start
- `HoldGridCanvas` — animated hold-grid hero background

**Design system:** CSS variables defined in `src/components/wall/styles.ts` as `GLOBAL_STYLES` string; injected via `<style>{GLOBAL_STYLES}</style>` per page. Fonts: Oswald (headers) + Space Mono (body). Primary color: `--cyan: #06b6d4`.

**Environment variables:** `VITE_API_URL` (default `/api/v1`), `VITE_CLERK_PUBLISHABLE_KEY`.

---

## Training Data

Model trained on Aurora/Kilter board climb data via the open-source [BoardLib API](https://github.com/lemeryfertitta/BoardLib). The DDPM approach is described in the [BetaZero v2 write-up](https://evmojo37.substack.com/p/betazero-v2-a-diffusion-model-for).

---

## Known Constraints / Future Work

- Generation is **synchronous** per request; an async job queue is planned
- SQLite is used for simplicity; no migration framework (manual `db_migration.py`)
- JSON arrays serialized as strings in SQLite (holds, edges, tags)
- CORS is open to all origins (development convenience)
- Cold-start latency on free-tier hosting handled client-side with 502 retry logic
