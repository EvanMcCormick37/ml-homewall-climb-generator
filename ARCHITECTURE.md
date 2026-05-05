# Architecture — BetaZero Climb Generator

BetaZero is an ML-powered climbing route generator. Users upload photos of their home walls, place holds on a canvas, and generate training climbs conditioned on grade and angle using a DDPM-based generative model.

For component-level detail see:
- [`climb-backend/ARCHITECTURE.md`](climb-backend/ARCHITECTURE.md) — FastAPI backend
- [`climb-frontend/ARCHITECTURE.md`](climb-frontend/ARCHITECTURE.md) — React frontend

---

## Repository Structure

```
ml-homewall-climb-generator/
├── climb-frontend/       # React + TypeScript frontend (production)
├── climb-backend/        # FastAPI Python backend (production)
├── model-training/       # ML model training scripts (offline, not in production)
├── cache/                # Reference/prototype frontend
├── tasks/                # Project planning docs
└── docker-compose.yaml   # Local dev setup
```

---

## System Overview

```
Browser
  │
  │  HTTPS
  ▼
climb-frontend  (React + TanStack Router + Clerk)
  │
  │  REST /api/v1/...
  │  Bearer JWT (Clerk RS256)
  ▼
climb-backend  (FastAPI + SQLite)
  │
  ├── data/layouts/{id}/photo.jpg   # Layout photos
  ├── data/storage.db               # SQLite database
  └── data/models/*.pth             # DDPM model weights
```

Authentication is handled entirely by **Clerk**: the frontend obtains a signed JWT, injects it into every API request, and the backend verifies it against Clerk's JWKS endpoint (cached 1 hour) without any session state.

---

## Frontend (`climb-frontend/`)

| Concern   | Library / Version                                        |
| --------- | -------------------------------------------------------- |
| Framework | React 19 + TypeScript 5.9                                |
| Build     | Vite 7.2                                                 |
| Routing   | TanStack Router 1.143 (file-based, auto-generated tree)  |
| Auth      | Clerk 5.61 (`@clerk/clerk-react`)                        |
| HTTP      | Axios 1.13 with Bearer token interceptor                 |
| Styling   | Tailwind CSS 4 + CSS variables (BetaZero design system)  |
| Icons     | lucide-react                                             |

### Routes

| Route              | File                      | Purpose                              |
| ------------------ | ------------------------- | ------------------------------------ |
| `/`                | `index.tsx`               | Homepage — layout cards, "Add Wall"  |
| `/signIn`          | `signIn.tsx`              | Clerk hosted sign-in                 |
| `/signUp`          | `signUp.tsx`              | Clerk hosted sign-up                 |
| `/layouts/new`     | `layouts/new.tsx`         | 4-step layout creation wizard        |
| `/$layoutId/holds` | `$layoutId/holds.tsx`     | Canvas-based hold placement editor   |
| `/$layoutId/set`   | `$layoutId/set.tsx`       | Climb generator (main feature)       |
| `/$layoutId/view`  | `$layoutId/view.tsx`      | Saved climb browser                  |
| `/$layoutId/sizes` | `$layoutId/sizes.tsx`     | Size variant manager                 |

### Design System

Global CSS variables defined in `src/components/wall/styles.ts` as `GLOBAL_STYLES`. Primary color: `--cyan: #06b6d4`. Background: `--bg: #09090b`. Fonts: **Oswald** (headers) and **Space Mono** (mono/body).

### Coordinate System

Holds are stored in **feet** (origin: bottom-left). The canvas renders in **pixels** (origin: top-left). `useHolds` converts between them — Y axis is inverted.

---

## Backend (`climb-backend/`)

| Concern    | Library / Version                             |
| ---------- | --------------------------------------------- |
| Framework  | FastAPI 0.125 (async)                         |
| Server     | Uvicorn 0.38                                  |
| Database   | SQLite 3 (raw SQL + `Row` factory)            |
| Auth       | Clerk JWT via PyJWT 2.11 + RSA key caching    |
| ML         | PyTorch 2.10 (CPU), scikit-learn 1.6, NumPy   |
| Config     | pydantic-settings (loaded from `.env`)        |

### API Endpoints (prefix: `/api/v1`)

#### Layouts — `/layouts`

| Method   | Path                       | Auth         | Description                                      |
| -------- | -------------------------- | ------------ | ------------------------------------------------ |
| `GET`    | `/`                        | Optional     | List public + user's own layouts                 |
| `GET`    | `/{layout_id}`             | Access check | Get layout metadata, holds, and sizes            |
| `POST`   | `/`                        | Required     | Create a new layout                              |
| `PUT`    | `/{layout_id}/edit`        | Owner        | Update name / description / visibility           |
| `PUT`    | `/{layout_id}/photo`       | Owner        | Upload or replace the layout photo               |
| `PUT`    | `/{layout_id}/holds`       | Owner        | Replace the entire hold set                      |
| `DELETE` | `/{layout_id}`             | Owner        | Delete layout (cascades to sizes, holds, climbs) |
| `GET`    | `/{layout_id}/photo`       | Access check | Download full-resolution photo                   |
| `GET`    | `/{layout_id}/photo-small` | Access check | Download 1/4-scale thumbnail                     |

#### Sizes — `/layouts/{layout_id}/sizes`

| Method   | Path         | Auth     | Description              |
| -------- | ------------ | -------- | ------------------------ |
| `GET`    | `/`          | Optional | List size variants       |
| `POST`   | `/`          | Required | Add a size variant       |
| `DELETE` | `/{size_id}` | Required | Delete a size variant    |

#### Climbs — `/layouts/{layout_id}/climbs`

| Method   | Path          | Auth     | Description              |
| -------- | ------------- | -------- | ------------------------ |
| `GET`    | `/`           | Optional | List climbs with filters |
| `POST`   | `/`           | Required | Save a single climb      |
| `POST`   | `/batch`      | Optional | Bulk insert climbs       |
| `DELETE` | `/{climb_id}` | Required | Delete a climb           |

#### Generation — `/layouts/{layout_id}/generate`

| Method | Path | Description                        |
| ------ | ---- | ---------------------------------- |
| `GET`  | `/`  | Generate climbs via the DDPM model |

Parameters: `num_climbs` (1–10), `grade`, `grade_scale`, `angle`, `timesteps`, `t_start_projection`, `x_offset`, `guidance_value`, `deterministic`, `seed`.

### Database Schema (SQLite)

Five tables: `users`, `layouts`, `sizes`, `holds`, `climbs`. JSON arrays (edges, holds, tags) are stored as serialized strings. Full schema in [`climb-backend/ARCHITECTURE.md`](climb-backend/ARCHITECTURE.md).

### Authentication

Clerk JWTs (RS256) verified against JWKS from `{CLERK_ISSUER}/.well-known/jwks.json` (cached 1 hour). FastAPI dependency functions in `auth.py` provide `get_current_user`, `require_auth`, `sync_auth`, `get_accessible_layout`, and `require_layout_owner`.

Layout visibility: `public` (anyone), `unlisted` (anyone with share token), `private` (owner or share token).

---

## Core Concepts

### Layout vs. Size

A **Layout** is a unique hold arrangement — it owns the wall photo and all hold positions. A **Size** is a physical dimension variant of that layout (e.g., the same board mounted as 8×10 or 12×12), defined by edge-crop bounds `[left, right, bottom, top]` in feet.

- Photos are owned by layouts, not sizes
- Holds are attached to layouts; sizes filter holds by their edge bounds
- A layout can have multiple sizes; each size has a `kickboard` flag

### ML Generation

The generative model is a **DDPM (Denoising Diffusion Probabilistic Model)** trained on Aurora board climb data. At generation time:

1. The backend loads the layout's holds from the database
2. The DDPM conditions on `grade` and `angle`, denoising random noise into a point cloud
3. A UNet hold classifier assigns roles — start / finish / hand / foot
4. Manifold guidance snaps generated points to the nearest real hold positions
5. The result is returned as a `Holdset` (hold indices, not coordinates)

The model is lazy-loaded into memory on first use via a thread-safe pool (default size: 4). Weights live in `data/models/`.

---

## Key Data Flows

### Create a Layout

1. `/layouts/new` wizard: Visibility → Upload photo → Crop → Details (name, dimensions, default angle)
2. On submit: `POST /layouts` → `PUT /layouts/{id}/photo` → `POST /layouts/{id}/sizes`
3. Redirect to `/$layoutId/holds` for hold placement

### Place Holds

- Canvas editor at `/$layoutId/holds` with modes: `add` / `remove` / `select` / `edit`
- On save: `PUT /layouts/{id}/holds` — replaces the entire hold array atomically

### Generate Climbs

- `/$layoutId/set` — select grade, angle, and generation settings
- `GET /layouts/{id}/generate?...` — synchronous model inference (~2–5 s), returns `Holdset[]`
- User navigates results, saves chosen climb via `POST /layouts/{id}/climbs`
- Climbs are shareable via URL-encoded hold indices (`sharing.ts`)

### Browse Saved Climbs

- `/$layoutId/view` — fetch climbs with filters (grade range, setter, tags, holds)
- Clicking a climb renders it on the wall canvas

---

## Local Development

**Backend**

```bash
cd climb-backend
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-lock.txt
# Create .env with CLERK_ISSUER and CLERK_SECRET_KEY
uvicorn app.main:app --reload
# Runs at http://localhost:8000
```

**Frontend**

```bash
cd climb-frontend
npm install
# Create .env with VITE_API_URL=http://localhost:8000/api/v1 and VITE_CLERK_PUBLISHABLE_KEY
npm run dev
# Runs at http://localhost:5173
```

Or use `docker-compose up` to bring up both services together.
