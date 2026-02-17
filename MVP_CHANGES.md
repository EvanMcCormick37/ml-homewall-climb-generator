# MVP Reconfiguration — Change Summary (Claude Opus 4.6)

## Files to DELETE (no longer needed)

### Backend

- `climb-backend/app/routers/climbs.py` — entire climbs router removed

### Frontend

- `climb-frontend/src/api/climbs.ts` — all climb API functions removed
- `climb-frontend/src/routes/walls/` — entire directory:
  - `climb-frontend/src/routes/walls/index.tsx`
  - `climb-frontend/src/routes/walls/new.tsx`
- `climb-frontend/src/routes/walls/$wallId/` — entire directory:
  - `climb-frontend/src/routes/walls/$wallId/index.tsx`
  - `climb-frontend/src/routes/walls/$wallId/view.tsx`
  - `climb-frontend/src/routes/walls/$wallId/holds.tsx`
  - `climb-frontend/src/routes/walls/$wallId/create.tsx`
  - `climb-frontend/src/routes/walls/$wallId/generate.tsx`

## Files MODIFIED (replace with versions in this output)

### Backend

| File                      | Change                                                                           |
| ------------------------- | -------------------------------------------------------------------------------- |
| `app/main.py`             | Removed climbs router include                                                    |
| `app/routers/__init__.py` | Removed climbs import                                                            |
| `app/routers/walls.py`    | Removed `create_wall`, `delete_wall`, `set_holds`, `upload_wall_photo` endpoints |

### Frontend

| File                    | Change                                                   |
| ----------------------- | -------------------------------------------------------- |
| `src/api/index.ts`      | Removed `climbs` re-export                               |
| `src/api/walls.ts`      | Removed `createWall`, `deleteWall`, `setHolds` functions |
| `src/hooks/useWalls.ts` | Removed `createNewWall`, `removeWall` from hook          |
| `src/routes/index.tsx`  | New homepage: wall card grid linking to `/$wallId`       |
| `src/routeTree.gen.ts`  | Regenerated with only `/` and `/$wallId` routes          |

## Files ADDED

| File                     | Purpose                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `src/routes/$wallId.tsx` | Generate page, moved from `walls/$wallId/generate.tsx` to `/$wallId` |

## Route Changes

| Before                                                                              | After                                          |
| ----------------------------------------------------------------------------------- | ---------------------------------------------- |
| `/` → Select Wall dropdown → `/walls/{id}` → Quick Actions → `/walls/{id}/generate` | `/` → Wall card grid → `/{id}` (generate page) |
| `/walls/new` (create wall)                                                          | **Removed**                                    |
| `/walls/{id}` (wall detail)                                                         | **Removed**                                    |
| `/walls/{id}/view` (view climbs)                                                    | **Removed**                                    |
| `/walls/{id}/holds` (edit holds)                                                    | **Removed**                                    |
| `/walls/{id}/create` (create climb)                                                 | **Removed**                                    |
| `/walls/{id}/generate`                                                              | `/{id}` (same page, new path)                  |

## API Endpoint Changes

| Endpoint                                 | Method | Status      |
| ---------------------------------------- | ------ | ----------- |
| `GET /api/v1/walls`                      | GET    | **Kept**    |
| `GET /api/v1/walls/{id}`                 | GET    | **Kept**    |
| `GET /api/v1/walls/{id}/photo`           | GET    | **Kept**    |
| `GET /api/v1/walls/{id}/generate`        | GET    | **Kept**    |
| `POST /api/v1/walls`                     | POST   | **Removed** |
| `DELETE /api/v1/walls/{id}`              | DELETE | **Removed** |
| `PUT /api/v1/walls/{id}/holds`           | PUT    | **Removed** |
| `PUT /api/v1/walls/{id}/photo`           | PUT    | **Removed** |
| `GET /api/v1/walls/{id}/climbs`          | GET    | **Removed** |
| `POST /api/v1/walls/{id}/climbs`         | POST   | **Removed** |
| `DELETE /api/v1/walls/{id}/climbs/{cid}` | DELETE | **Removed** |

## Notes

- After replacing files, run TanStack Router's code generation (`npx tsr generate`) to regenerate `routeTree.gen.ts` from the file system — or use the provided version directly.
- The `climb-backend/app/routers/generate.py` file is unchanged.
- Schema files, services, and database layer are untouched.
