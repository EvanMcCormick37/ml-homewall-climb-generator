# Architecture — BetaZero Climb Generator

BetaZero is an ML-powered climbing route generator. Users upload photos of their home walls, place holds, and generate training climbs conditioned on grade and angle using a DDPM-based generative model.

---

## Repository Structure

```
ml-homewall-climb-generator/
├── climb-frontend/       # React + TypeScript frontend
├── climb-backend/        # FastAPI Python backend
├── model-training/       # ML model training scripts (offline, not in production)
├── cache/                # Reference/prototype frontend
├── tasks/                # Project planning docs
└── docker-compose.yaml   # Local dev setup
```

---

## Frontend (`climb-frontend/`)

### Tech Stack

| Concern   | Library                                                 |
| --------- | ------------------------------------------------------- |
| Framework | React 19 + TypeScript 5.9                               |
| Build     | Vite 7.2                                                |
| Routing   | TanStack Router 1.143 (file-based)                      |
| Auth      | Clerk (`@clerk/clerk-react` 5.61)                       |
| HTTP      | Axios 1.13 with Bearer token interceptor                |
| Styling   | Tailwind CSS 4 + CSS variables (BetaZero design system) |
| Icons     | lucide-react                                            |

### Design System

Global CSS variables are defined in `src/components/wall/styles.ts` as the `GLOBAL_STYLES` string. Each page that needs them must include `<style>{GLOBAL_STYLES}</style>`.

```css
--cyan: #06b6d4; /* Primary action */
--cyan-dim: rgba(6, 182, 212, 0.15);
--bg: #09090b; /* Main background */
--surface: #111113; /* Card surface */
--surface2: #1c1c1e; /* Elevated surface */
--border: rgba(255, 255, 255, 0.08);
--border-active: rgba(6, 182, 212, 0.4);
--text-primary: #f4f4f5;
--text-muted: #71717a;
--text-dim: #52525b;
--radius: 4px;
```

Fonts: **Oswald** (headers, `.bz-oswald`) and **Space Mono** (body/mono, `.bz-mono`) via Google Fonts.

### Route Structure

File-based routing via TanStack Router (`src/routes/`):

| Route              | File                  | Purpose                                    |
| ------------------ | --------------------- | ------------------------------------------ |
| `/`                | `index.tsx`           | Homepage — layout cards, "Add Wall" button |
| `/signIn`          | `signIn.tsx`          | Clerk sign-in page                         |
| `/signUp`          | `signUp.tsx`          | Clerk sign-up page                         |
| `/layouts/new`     | `layouts/new.tsx`     | 4-step layout creation wizard              |
| `/$layoutId/holds` | `$layoutId/holds.tsx` | Canvas-based holds editor                  |
| `/$layoutId/set`   | `$layoutId/set.tsx`   | Climb generator (main feature)             |
| `/$layoutId/view`  | `$layoutId/view.tsx`  | Saved climb browser                        |
| `/$layoutId/sizes` | `$layoutId/sizes.tsx` | Size variant manager                       |
| (root)             | `__root.tsx`          | ClerkProvider + AuthInit wrapper           |

### API Client

`src/api/client.ts` — Axios instance pointed at `VITE_API_URL`. A request interceptor injects the Clerk JWT as a `Bearer` token on every outgoing request via `setAuthTokenProvider`.

API functions are organized by resource:

**`src/api/layouts.ts`**

- `getLayouts()` — `GET /layouts`
- `getLayout(id, sizeId?)` — `GET /layouts/{id}`
- `createLayout(data)` — `POST /layouts`
- `updateLayout(id, data)` — `PUT /layouts/{id}/edit`
- `uploadLayoutPhoto(id, file)` — `PUT /layouts/{id}/photo`
- `deleteLayout(id)` — `DELETE /layouts/{id}`
- `setLayoutHolds(id, holds)` — `PUT /layouts/{id}/holds`
- `getLayoutPhotoUrl(id)` — Returns a static URL string (no auth required for public layouts)

**`src/api/sizes.ts`**

- `getSizes(layoutId)` — `GET /layouts/{id}/sizes`
- `createSize(layoutId, data)` — `POST /layouts/{id}/sizes`
- `deleteSize(layoutId, sizeId)` — `DELETE /layouts/{id}/sizes/{sizeId}`

**`src/api/climbs.ts`**

- `getClimbs(wallId, filters)` — `GET /walls/{id}/climbs?...`
- `createClimb(wallId, data)` — `POST /walls/{id}/climbs`
- `deleteClimb(wallId, climbId)` — `DELETE /walls/{id}/climbs/{climbId}`

**`src/api/generate.ts`**

- `generateClimbs(wallId, request, settings?)` — `GET /walls/{id}/generate?...`

### Types

**`src/types/layout.ts`** — `LayoutMetadata`, `LayoutDetail`, `SizeMetadata`, `LayoutCreate`, `SizeCreate`

- `LayoutMetadata.dimensions: number[]` — `[width_ft, height_ft]`
- `SizeMetadata.edges: number[]` — `[left, right, bottom, top]` in feet

**`src/types/wall.ts`** — `HoldDetail`, `HoldMode`, `Visibility`, `Tag`

- `HoldMode = "add" | "remove" | "select" | "edit"`
- `HoldDetail.x/y` — position in feet; `is_foot` — `0`=hand hold, `1`=foot hold

**`src/types/climb.ts`** — `Climb`, `Holdset` (`{ start, finish, hand, foot }` as `number[]` of hold indices)

**`src/types/generate.ts`** — `GenerateRequest`, `GenerateSettings`, `GenerateResponse`

### Hooks

| Hook                            | File                    | Purpose                                       |
| ------------------------------- | ----------------------- | --------------------------------------------- |
| `useLayouts()`                  | `hooks/useLayouts.ts`   | Fetch all layouts, with 502 wake retry        |
| `useLayout(id, sizeId?)`        | `hooks/useLayouts.ts`   | Fetch a single layout                         |
| `useHolds(imageDims, wallDims)` | `hooks/useHolds.ts`     | Hold state + pixel↔feet coordinate conversion |
| `useClimbs()`                   | `hooks/useClimbs.ts`    | Fetch and filter saved climbs                 |
| `useImageCrop()`                | `hooks/useImageCrop.ts` | Crop state for the layout creation wizard     |
| `useAuth()`                     | `hooks/useAuth.ts`      | Clerk JWT token retrieval                     |

`fetchWithWakeRetry()` (defined in `useLayouts.ts`) retries failed requests for up to 20 seconds on 502 responses and surfaces a "Waking..." UI state — handling cold starts on free-tier hosting.

### Components

**`src/components/wall/`** — Core climb visualization

- `WallCanvas.tsx` — Renders the layout photo, hold markers, and active climb overlay
- `DisplaySettingsPanel.tsx` — UI for filtering which holds/climbs are shown
- `SaveShareMenu.tsx` — Save a climb + copy a share link
- `MobileSwipeNav.tsx` — Swipe between climbs on mobile
- `sharing.ts` — Encode/decode climbs as URL parameters for link sharing
- `styles.ts` — `GLOBAL_STYLES` CSS variable definitions

**Other components**

- `ImageCropper.tsx` — Crop UI with 8 draggable resize handles (used in `/layouts/new`)
- `HoldFeaturesSidebar`, `EnabledFeaturesMenu` — Hold property editing panels
- `screens.tsx` — `WakingScreen` loading state

### Coordinate System

Holds are stored in **feet** (origin: bottom-left of wall). The canvas displays them in **pixels** (origin: top-left of image). `useHolds` converts between them:

```
toFeetCoords:   x = pixelX / imgWidth * wallWidth
                y = (imgHeight - pixelY) / imgHeight * wallHeight   ← Y axis is inverted

toPixelCoords:  px = hold.x / wallWidth * imgWidth
                py = imgHeight - hold.y / wallHeight * imgHeight
```

---

## Backend (`climb-backend/`)

### Tech Stack

| Concern   | Library                               |
| --------- | ------------------------------------- |
| Framework | FastAPI 0.125 (async)                 |
| Database  | SQLite 3 (raw SQL + Row factory)      |
| Auth      | Clerk JWT via PyJWT + RSA key caching |
| ML        | PyTorch 2.10 (CPU), scikit-learn      |
| Server    | Uvicorn 0.38                          |

### Application Structure

```
app/
├── main.py              # FastAPI app, router registration, startup hook
├── config.py            # Settings via pydantic-settings (.env)
├── database.py          # SQLite schema init and migrations
├── auth.py              # Clerk JWT verification + FastAPI access control deps
├── routers/
│   ├── layouts.py
│   ├── sizes.py
│   ├── climbs.py
│   └── generate.py
├── schemas/
│   ├── base.py          # HoldDetail, Holdset
│   ├── layouts.py       # LayoutMetadata, LayoutDetail, LayoutCreate
│   ├── sizes.py         # SizeMetadata, SizeCreate
│   ├── climbs.py        # Climb, ClimbCreate, ClimbListResponse
│   └── generate.py      # GenerateRequest, GenerateSettings, GenerateResponse
└── services/
    ├── container.py     # ServiceContainer dataclass (dependency injection)
    ├── layout_service.py
    ├── size_service.py
    ├── climb_service.py
    ├── generation_service.py
    ├── user_service.py
    └── utils/
        ├── conversion_utils.py  # SQLite Row ↔ Pydantic schema helpers
        └── generation_utils.py  # Grade/angle lookups + Generator class wrapper
```

### Database Schema

SQLite file at `data/storage.db`. JSON arrays (holds, edges, tags) are serialized as strings.

```sql
CREATE TABLE users (
    id           TEXT PRIMARY KEY,
    email        TEXT NOT NULL,
    display_name TEXT,
    avatar_url   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE layouts (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    dimensions    TEXT,          -- JSON: [width_ft, height_ft]
    image_edges   TEXT,          -- JSON: [left, right, bottom, top] in feet
    default_angle INTEGER,
    owner_id      TEXT,
    visibility    TEXT DEFAULT 'public',  -- 'public' | 'private' | 'unlisted'
    share_token   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sizes (
    id         TEXT PRIMARY KEY,
    layout_id  TEXT NOT NULL REFERENCES layouts(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    edges      TEXT NOT NULL,  -- JSON: [left, right, bottom, top] in feet
    kickboard  BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE holds (
    id          TEXT PRIMARY KEY,
    layout_id   TEXT,
    hold_index  INTEGER NOT NULL,   -- unique per layout
    x           REAL NOT NULL,      -- feet from left
    y           REAL NOT NULL,      -- feet from bottom
    pull_x      REAL DEFAULT 0.0,   -- normalized pull direction [-1, 1]
    pull_y      REAL DEFAULT 0.0,
    useability  REAL DEFAULT 0.5,   -- [0, 1]
    tags        TEXT,               -- JSON array of tag strings
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE climbs (
    id          TEXT PRIMARY KEY,
    layout_id   TEXT,               -- canonical FK
    angle       INTEGER NOT NULL,
    name        TEXT NOT NULL,
    holds       TEXT NOT NULL,      -- JSON Holdset: {start, finish, hand, foot}
    tags        TEXT,               -- JSON array of tag strings
    grade       REAL,               -- internal difficulty float
    quality     REAL DEFAULT 2.5,   -- [0, 3]
    ascents     INTEGER DEFAULT 0,
    setter_name TEXT,
    setter_id   TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE
);
```

### API Endpoints

All routes are prefixed with `/api/v1`.

#### Layouts — `/api/v1/layouts`

| Method   | Path                 | Auth         | Description                                      |
| -------- | -------------------- | ------------ | ------------------------------------------------ |
| `GET`    | `/`                  | Optional     | List public + user's own layouts                 |
| `GET`    | `/{layout_id}`       | Access check | Get layout with holds and sizes                  |
| `POST`   | `/`                  | Required     | Create a new layout (metadata only, no photo)    |
| `PUT`    | `/{layout_id}/edit`  | Owner        | Update name / description / visibility           |
| `PUT`    | `/{layout_id}/photo` | Owner        | Upload or replace the layout photo               |
| `PUT`    | `/{layout_id}/holds` | Owner        | Replace the entire hold set                      |
| `DELETE` | `/{layout_id}`       | Owner        | Delete layout (cascades to sizes, holds, climbs) |
| `GET`    | `/{layout_id}/photo` | Access check | Download the layout photo                        |

#### Sizes — `/api/v1/layouts/{layout_id}/sizes`

| Method   | Path         | Auth         | Description                         |
| -------- | ------------ | ------------ | ----------------------------------- |
| `GET`    | `/`          | Access check | List all size variants for a layout |
| `POST`   | `/`          | Required     | Add a new size variant              |
| `DELETE` | `/{size_id}` | Required     | Delete a size variant               |

#### Climbs — `/api/v1/layouts/{layout_id}/climbs`

| Method   | Path          | Auth     | Description              |
| -------- | ------------- | -------- | ------------------------ |
| `GET`    | `/`           | Optional | List climbs with filters |
| `POST`   | `/`           | Required | Save a single climb      |
| `POST`   | `/batch`      | Optional | Bulk insert climbs       |
| `DELETE` | `/{climb_id}` | Required | Delete a climb           |

Supported query parameters for `GET /climbs`: `angle`, `grade_scale`, `min_grade`, `max_grade`, `include_projects`, `setter_name`, `name_includes`, `holds_include[]`, `tags_include[]`, `after`, `sort_by`, `descending`, `limit`, `offset`.

Grades are stored internally as floats via a `GRADE_TO_DIFF` map (e.g. `"V4"` → `8.0`, `"6b+"` → `17.5`). Supported scales: `v_grade` (V0–V17) and `font` (1a–9a+).

#### Generation — `/api/v1/layouts/{layout_id}/generate`

| Method | Path | Description                        |
| ------ | ---- | ---------------------------------- |
| `GET`  | `/`  | Generate climbs via the DDPM model |

Query parameters: `num_climbs` (1–10), `grade`, `grade_scale`, `angle`, `timesteps`, `t_start_projection`, `x_offset`, `deterministic`, `seed`.

Response: `{ wall_id, climbs: Holdset[], num_generated, parameters }`.

> Generation is currently **synchronous** — the model runs inline per request (~2–5 s). An async job queue is planned for a future phase.

#### Misc

| Method | Path      | Description                              |
| ------ | --------- | ---------------------------------------- |
| `GET`  | `/health` | Health check — `{ "status": "healthy" }` |

### Authentication

Clerk JWTs (RS256) are verified against JWKS fetched from `{CLERK_ISSUER}/.well-known/jwks.json`, cached for 1 hour. The token subject (`sub`) is the user ID.

FastAPI dependency functions in `auth.py`:

| Dependency              | Behavior                                                              |
| ----------------------- | --------------------------------------------------------------------- |
| `get_current_user`      | Extracts and verifies the Bearer token; returns a user dict or `None` |
| `require_auth`          | 401 if not authenticated                                              |
| `sync_auth`             | Same as `require_auth` but also upserts the user into the DB          |
| `get_accessible_layout` | 404 if not found; 403 if the caller has no access                     |
| `require_layout_owner`  | 401 if not authenticated; 403 if not the owner                        |

**Layout access rules:**

- `public` → anyone can read
- `unlisted` → anyone with `?share_token=` in the request URL
- `private` → owner only, or a valid `share_token`

On first auth, the user is upserted into the `users` table (email, display_name, avatar_url synced from Clerk).

### Services Layer

Business logic lives in `services/`. Route handlers receive a `ServiceContainer` via FastAPI's dependency injection and call service functions — no raw SQL in routers.

**`layout_service.py`** — Layout and hold CRUD; photo storage at `data/layouts/{layout_id}`

**`size_service.py`** — Size CRUD

**`climb_service.py`** — Climb CRUD with rich filtering; batch insert; `Holdset` validation

**`generation_service.py`** — Lazy-loads the DDPM model on first call; resolves the angle from the request or layout default; dispatches to the `Generator` class in `utils/generation_utils.py`; converts raw model output to `Holdset` format

**`user_service.py`** — `ensure_user_exists(user_id, email, name)` upsert

### Configuration

`app/config.py` uses `pydantic-settings`, loading from `.env`:

```
CLERK_ISSUER          Clerk issuer URL (required)
CLERK_SECRET_KEY      Clerk secret key (required)
DATA_DIR              data/
WALLS_DIR             data/walls/       (legacy photo storage)
LAYOUTS_DIR           data/layouts/     (current photo storage)
DB_PATH               data/storage.db
DDPM_WEIGHTS_PATH     data/models/ddpm-weights.pth
SCALER_WEIGHTS_PATH   data/models/scaler-weights.joblib
HC_WEIGHTS_PATH       data/models/unet-hold-classifier.pth
LIMIT                 50  (default pagination limit)
```

---

## Key Architectural Concepts

### Layout vs. Size

A **Layout** is a unique hold arrangement — it owns the wall photo and all hold positions. A **Size** is a physical dimension variant of that layout (e.g., the same board mounted as 8×10 or 12×12), defined by edge-crop bounds `[left, right, bottom, top]` in feet.

- Photos are owned by layouts, not by sizes
- Holds are attached to layouts; sizes filter holds by their edge bounds
- A layout can have multiple sizes; each size can set a `kickboard` flag

### Walls → Layouts Migration

The original data model used a `walls` table. It has been replaced by `layouts` + `sizes`, but the old table is preserved for backward compatibility. `wall_id` and `layout_id` always hold the same value for migrated data. The legacy `walls` table and `/api/v1/walls/` endpoints will be removed in a future cleanup phase.

### ML Generation

The generative model is a **DDPM (Denoising Diffusion Probabilistic Model)** trained on Aurora board climb data. At generation time:

1. The backend loads the layout's holds from the database
2. The DDPM conditions on `grade` and `angle`, denoising random noise into a point cloud
3. A hold classifier (UNet) assigns roles — start/finish/hand/foot — to points
4. Manifold guidance snaps generated points to the nearest real hold positions
5. The result is returned as a `Holdset` (hold indices, not coordinates)

The model is lazy-loaded into memory on first use and kept resident. Weights live in `data/models/`.

---

## Data Flow: Key User Journeys

### Create a Layout

1. `/layouts/new` — 4-step wizard: Visibility → Upload photo → Crop → Details (name, dimensions, default angle, kickboard)
2. On submit: `POST /layouts` → `PUT /layouts/{id}/photo` → `POST /layouts/{id}/sizes`
3. Redirects to `/$layoutId/holds` for hold placement

### Place Holds

- Canvas editor at `/$layoutId/holds`
- Interaction modes: `add` / `remove` / `select` / `edit`
- On save: `PUT /layouts/{id}/holds` — replaces the entire hold array atomically

### Generate Climbs

- `/$layoutId/set` — user selects grade, angle, and generation settings (timesteps, etc.)
- `GET /layouts/{id}/generate?...` — synchronous model inference, returns `Holdset[]`
- User navigates results, then saves a chosen climb via `POST /layouts/{id}/climbs`
- Climbs are shareable via URL-encoded hold indices (`sharing.ts`)

### Browse Saved Climbs

- `/$layoutId/view` — fetches climbs with optional filters (grade range, setter, tags, holds)
- Clicking a climb visualizes it on the wall canvas

---

## Local Development

**Backend**

```bash
cd climb-backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Create .env with CLERK_ISSUER and CLERK_SECRET_KEY
uvicorn app.main:app --reload
# Runs at http://localhost:8000
```

**Frontend**

```bash
cd climb-frontend
npm install
# Create .env with VITE_API_URL and VITE_CLERK_PUBLISHABLE_KEY
npm run dev
# Runs at http://localhost:5173
```

Or use `docker-compose up` to bring up both services together.
