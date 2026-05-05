# Architecture — climb-backend

FastAPI backend for BetaZero. Handles layout/hold/climb CRUD, layout photo storage, Clerk JWT authentication, and ML-based climb generation via a pre-trained DDPM model.

---

## Tech Stack

| Concern    | Library / Version                            |
| ---------- | -------------------------------------------- |
| Framework  | FastAPI 0.125 (async)                        |
| Server     | Uvicorn 0.38                                 |
| Database   | SQLite 3 (raw SQL + `Row` factory)           |
| Auth       | Clerk JWT via PyJWT 2.11 + RSA key caching   |
| ML         | PyTorch 2.10 (CPU), scikit-learn 1.6, NumPy  |
| Config     | pydantic-settings 2.x (loaded from `.env`)   |
| HTTP (out) | httpx 0.28 (async, for JWKS fetching)        |

---

## Directory Structure

```
climb-backend/
├── app/
│   ├── main.py                    # FastAPI app, router registration, startup hook
│   ├── config.py                  # Settings via pydantic-settings (.env)
│   ├── database.py                # SQLite schema init + migration helpers
│   ├── auth.py                    # Clerk JWT verification + FastAPI access-control deps
│   ├── routers/
│   │   ├── layouts.py             # Layout + photo + holds endpoints
│   │   ├── sizes.py               # Size variant endpoints
│   │   ├── climbs.py              # Climb CRUD + batch insert
│   │   └── generate.py            # ML generation endpoint
│   ├── schemas/
│   │   ├── base.py                # HoldDetail, Holdset
│   │   ├── layouts.py             # LayoutMetadata, LayoutDetail, LayoutCreate
│   │   ├── sizes.py               # SizeMetadata, SizeCreate
│   │   ├── climbs.py              # Climb, ClimbCreate, ClimbListResponse
│   │   └── generate.py            # GenerateRequest, GenerateSettings, GenerateResponse
│   ├── services/
│   │   ├── container.py           # ServiceContainer dataclass (DI root)
│   │   ├── layout_service.py      # Layout + hold CRUD, photo storage
│   │   ├── size_service.py        # Size CRUD
│   │   ├── climb_service.py       # Climb CRUD + rich filtering
│   │   ├── generation_service.py  # DDPM generation orchestration
│   │   ├── user_service.py        # ensure_user_exists upsert
│   │   └── utils/
│   │       ├── conversion_utils.py # SQLite Row ↔ Pydantic + grade/difficulty maps
│   │       └── generation_utils.py # Generator class wrapper + pool management
│   └── test/                      # pytest test suite
├── db_migration.py                # Manual migration runner
├── requirements-lock.txt          # Pinned dependencies
└── data/                          # Runtime data (excluded from git)
    ├── storage.db                 # SQLite database
    ├── layouts/{layout_id}/       # Layout photos (photo.jpg, photo-small.jpg)
    └── models/                    # ML model weights
```

---

## Startup

`app/main.py` creates the FastAPI app, registers CORS middleware (all origins), mounts all routers under `/api/v1`, and calls `init_db()` in a startup event to create/migrate the SQLite schema.

---

## Database Schema

SQLite file at `data/storage.db`. JSON arrays (holds, edges, tags) are stored as serialized strings.

```sql
CREATE TABLE users (
    id           TEXT PRIMARY KEY,       -- Clerk user ID
    email        TEXT NOT NULL,
    display_name TEXT,
    avatar_url   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE layouts (
    id                      TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    description             TEXT,
    dimensions              TEXT,          -- JSON: [width_ft, height_ft]
    image_edges             TEXT,          -- JSON: [left, right, bottom, top] in feet
    homography_src_corners  TEXT,          -- JSON: [tlx,tly, trx,try, blx,bly, brx,bry] for trapezoid mapping (nullable)
    default_angle           INTEGER,
    owner_id                TEXT,
    visibility              TEXT DEFAULT 'public',   -- 'public' | 'private' | 'unlisted'
    share_token             TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sizes (
    id         TEXT PRIMARY KEY,
    layout_id  TEXT NOT NULL REFERENCES layouts(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    edges      TEXT NOT NULL,   -- JSON: [left, right, bottom, top] in feet
    kickboard  BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE holds (
    id          TEXT PRIMARY KEY,
    layout_id   TEXT,
    hold_index  INTEGER NOT NULL,   -- ordinal index, unique per layout
    x           REAL NOT NULL,      -- feet from left edge
    y           REAL NOT NULL,      -- feet from bottom edge
    pull_x      REAL DEFAULT 0.0,   -- normalized pull direction [-1, 1]
    pull_y      REAL DEFAULT 0.0,
    useability  REAL DEFAULT 0.5,   -- quality rating [0, 1]
    is_foot     BOOLEAN DEFAULT 0,  -- 0 = hand hold, 1 = foot hold
    tags        TEXT,               -- JSON array of tag strings (e.g. "pinch", "flat")
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE climbs (
    id          TEXT PRIMARY KEY,
    layout_id   TEXT,
    angle       INTEGER NOT NULL,
    name        TEXT NOT NULL,
    holds       TEXT NOT NULL,      -- JSON Holdset: {start, finish, hand, foot} as hold-index arrays
    tags        TEXT,               -- JSON array of tag strings
    grade       REAL,               -- internal difficulty float (see GRADE_TO_DIFF)
    quality     REAL DEFAULT 2.5,   -- [0, 3]
    ascents     INTEGER DEFAULT 0,
    setter_name TEXT,
    setter_id   TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE
);
```

---

## API Endpoints

All routes are prefixed with `/api/v1`.

### Layouts — `/api/v1/layouts`

| Method   | Path                       | Auth         | Description                                        |
| -------- | -------------------------- | ------------ | -------------------------------------------------- |
| `GET`    | `/`                        | Optional     | List public layouts + authenticated user's layouts |
| `GET`    | `/{layout_id}`             | Access check | Get layout metadata, holds, and sizes              |
| `POST`   | `/`                        | Required     | Create a new layout                                |
| `PUT`    | `/{layout_id}/edit`        | Owner        | Update name / description / visibility / angle     |
| `PUT`    | `/{layout_id}/photo`       | Owner        | Upload or replace the layout photo                 |
| `PUT`    | `/{layout_id}/holds`       | Owner        | Replace the entire hold set atomically             |
| `DELETE` | `/{layout_id}`             | Owner        | Delete layout (cascades to sizes, holds, climbs)   |
| `GET`    | `/{layout_id}/photo`       | Access check | Download the full-resolution layout photo          |
| `GET`    | `/{layout_id}/photo-small` | Access check | Download the 1/4-scale thumbnail photo             |

### Sizes — `/api/v1/layouts/{layout_id}/sizes`

| Method   | Path         | Auth         | Description                        |
| -------- | ------------ | ------------ | ---------------------------------- |
| `GET`    | `/`          | Access check | List all size variants for a layout |
| `POST`   | `/`          | Required     | Add a new size variant             |
| `DELETE` | `/{size_id}` | Required     | Delete a size variant              |

### Climbs — `/api/v1/layouts/{layout_id}/climbs`

| Method   | Path          | Auth     | Description            |
| -------- | ------------- | -------- | ---------------------- |
| `GET`    | `/`           | Optional | List climbs with filters |
| `POST`   | `/`           | Required | Save a single climb    |
| `POST`   | `/batch`      | Optional | Bulk insert climbs     |
| `DELETE` | `/{climb_id}` | Required | Delete a climb         |

Supported query parameters for `GET /climbs`: `angle`, `grade_scale`, `min_grade`, `max_grade`, `include_projects`, `setter_name`, `name_includes`, `holds_include[]`, `tags_include[]`, `after`, `sort_by`, `descending`, `limit`, `offset`.

Grades are stored as floats via a `GRADE_TO_DIFF` map (e.g. `"V4"` → `8.0`, `"6b+"` → `17.5`). Supported scales: `v_grade` (V0–V17) and `font` (1a–9a+).

### Generation — `/api/v1/layouts/{layout_id}/generate`

| Method | Path | Description                        |
| ------ | ---- | ---------------------------------- |
| `GET`  | `/`  | Generate climbs via the DDPM model |

Query parameters: `num_climbs` (1–10), `grade`, `grade_scale`, `angle`, `timesteps`, `t_start_projection`, `x_offset`, `guidance_value`, `deterministic`, `seed`.

Response: `{ wall_id, climbs: Holdset[], num_generated, parameters }`.

Generation runs **synchronously** inline per request (~2–5 s). An async job queue is planned for a future phase.

### Misc

| Method | Path      | Description                              |
| ------ | --------- | ---------------------------------------- |
| `GET`  | `/health` | Health check — `{ "status": "healthy" }` |

---

## Authentication

Clerk JWTs (RS256) are verified against JWKS fetched from `{CLERK_ISSUER}/.well-known/jwks.json`, cached for 1 hour. The token subject (`sub`) is the Clerk user ID.

FastAPI dependency functions in `auth.py`:

| Dependency              | Behavior                                                               |
| ----------------------- | ---------------------------------------------------------------------- |
| `get_current_user`      | Extracts and verifies the Bearer token; returns a user dict or `None`  |
| `require_auth`          | 401 if not authenticated                                               |
| `sync_auth`             | Same as `require_auth` but also upserts the user into the `users` table |
| `get_accessible_layout` | 404 if not found; 403 if the caller lacks access                       |
| `require_layout_owner`  | 401 if not authenticated; 403 if not the layout owner                  |

Layout access rules:

- `public` → anyone can read
- `unlisted` → anyone with `?share_token=` in the request URL
- `private` → owner only, or with a valid `share_token`

---

## Services Layer

Business logic lives in `services/`. Route handlers receive a `ServiceContainer` via FastAPI dependency injection and call service methods — no raw SQL in routers.

| Service                 | Responsibility                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `layout_service.py`     | Layout + hold CRUD; photo storage at `data/layouts/{layout_id}/`                   |
| `size_service.py`       | Size CRUD                                                                           |
| `climb_service.py`      | Climb CRUD with rich filtering; batch insert; Holdset serialization                 |
| `generation_service.py` | Acquires a generator from the pool; resolves angle; converts model output to Holdset |
| `user_service.py`       | `ensure_user_exists(user_id, email, name)` upsert on every authenticated request    |

---

## ML Generation

The generative model is a **DDPM (Denoising Diffusion Probabilistic Model)** trained on Aurora board climb data.

**Weights** (in `data/models/`):
- `ddpm-weights.pth` — core diffusion model
- `scaler-weights.joblib` — feature scaler
- `unet-hold-classifier.pth` — hold role classifier (UNet)

**Generation pipeline:**
1. Load the layout's hold positions from the database
2. DDPM conditions on `grade` and `angle`, denoising random noise into a point cloud
3. The UNet hold classifier assigns roles — start / finish / hand / foot
4. Manifold guidance snaps generated points to the nearest real hold positions
5. Result is returned as a `Holdset` (hold indices, not coordinates)

A thread-safe `Generator` pool (default size: 4) allows concurrent requests without re-initializing model weights on each call. Models are lazy-loaded on first generation request.

---

## File Storage

```
data/
├── storage.db                    # SQLite database
├── layouts/{layout_id}/
│   ├── photo.jpg (or .png)       # Full-resolution layout photo
│   └── photo-small.jpg (or .png) # 1/4-scale thumbnail
└── models/
    ├── ddpm-weights.pth
    ├── scaler-weights.joblib
    └── unet-hold-classifier.pth
```

---

## Configuration

`app/config.py` loads from `.env` via `pydantic-settings`:

```
CLERK_ISSUER           Clerk issuer URL (required)
CLERK_SECRET_KEY       Clerk secret key (required)
DATA_DIR               data/
LAYOUTS_DIR            data/layouts/
DB_PATH                data/storage.db
DDPM_WEIGHTS_PATH      data/models/ddpm-weights.pth
SCALER_WEIGHTS_PATH    data/models/scaler-weights.joblib
HC_WEIGHTS_PATH        data/models/unet-hold-classifier.pth
GENERATOR_POOL_SIZE    4   (concurrent generation slots)
LIMIT                  50  (default pagination limit)
```

---

## Local Development

```bash
cd climb-backend
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-lock.txt
# Create .env with CLERK_ISSUER and CLERK_SECRET_KEY
uvicorn app.main:app --reload
# Runs at http://localhost:8000
```
