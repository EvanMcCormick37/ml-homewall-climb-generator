# Climb Generator API

A REST API for managing climbing walls, climbs, and ML-based climb generation.

## Project Structure

```
climb-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── database.py          # SQLite setup and connection management
│   ├── routers/             # API endpoint definitions
│   │   ├── walls.py
│   │   ├── climbs.py
│   │   ├── models.py
│   │   └── jobs.py
│   ├── schemas/             # Pydantic request/response models
│   │   ├── walls.py
│   │   ├── climbs.py
│   │   ├── models.py
│   │   └── jobs.py
│   └── services/            # Business logic
│       ├── wall_service.py
│       ├── climb_service.py
│       ├── model_service.py
│       └── job_service.py
├── data/                    # Created at runtime
│   ├── climbs.db           # SQLite database
│   └── walls/              # Wall data directories
│       └── {wall-id}/
│           ├── wall.json   # Hold definitions
│           ├── photo.jpg   # Wall photo
│           └── models/     # Trained model weights
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Walls

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/walls` | List all walls |
| POST | `/walls` | Create a new wall |
| GET | `/walls/{wall_id}` | Get wall details with holds |
| DELETE | `/walls/{wall_id}` | Delete wall and all data |
| GET | `/walls/{wall_id}/photo` | Get wall photo |
| PUT | `/walls/{wall_id}/photo` | Upload wall photo |

### Climbs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/walls/{wall_id}/climbs` | List climbs (with filters) |
| POST | `/walls/{wall_id}/climbs` | Create a climb |
| DELETE | `/walls/{wall_id}/climbs/{climb_id}` | Delete a climb |

**Climb Query Parameters:**
- `name` - Filter by name (partial match)
- `setter` - Filter by setter ID
- `after` - Climbs created after date
- `sort_by` - Sort by: date, name, grade, num_holds
- `limit` / `offset` - Pagination
- `includes_holds` - Comma-separated hold IDs that must be in climb

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/walls/{wall_id}/models` | List models |
| POST | `/walls/{wall_id}/models` | Create and train model |
| GET | `/walls/{wall_id}/models/{model_id}` | Get model details |
| DELETE | `/walls/{wall_id}/models/{model_id}` | Delete model |
| POST | `/walls/{wall_id}/models/{model_id}/generate` | Generate climbs |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/jobs/{job_id}` | Get background job status |

## Background Jobs

Model training runs as a background task. The workflow:

1. `POST /walls/{wall_id}/models` returns immediately with `job_id`
2. Client polls `GET /jobs/{job_id}` for status
3. Job progresses: `PENDING` → `PROCESSING` → `COMPLETED` (or `FAILED`)
4. Once `COMPLETED`, model is ready for generation

## Example Usage

### Create a wall with holds

```bash
curl -X POST http://localhost:8000/walls \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Home Wall",
    "holds": [
      {"hold_id": 1, "norm_x": 0.2, "norm_y": 0.1, "pull_x": 0, "pull_y": 1, "useability": 8},
      {"hold_id": 2, "norm_x": 0.5, "norm_y": 0.3, "pull_x": 0.5, "pull_y": 0.5, "useability": 5}
    ]
  }'
```

### Add a climb

```bash
curl -X POST http://localhost:8000/walls/{wall_id}/climbs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Climb",
    "grade": "V3",
    "sequence": [[1, 2], [3, 2], [3, 4], [5, 4]]
  }'
```

### Train a model

```bash
# Start training
curl -X POST http://localhost:8000/walls/{wall_id}/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "mlp",
    "epochs": 100,
    "augment_dataset": true
  }'
# Returns: {"model_id": "...", "job_id": "..."}

# Poll for status
curl http://localhost:8000/jobs/{job_id}
# Returns: {"status": "PROCESSING", "progress": 0.45, ...}
```

### Generate climbs

```bash
curl -X POST http://localhost:8000/walls/{wall_id}/models/{model_id}/generate \
  -H "Content-Type: application/json" \
  -d '{
    "starting_holds": [1, 2],
    "max_moves": 10,
    "num_climbs": 5,
    "temperature": 1.0
  }'
```

## Development Status

This is a skeleton implementation. The following still need to be implemented:

- [ ] `WallService.get_all_walls()`
- [ ] `WallService.get_wall()`
- [ ] `WallService.delete_wall()`
- [ ] `ClimbService.get_climbs()` - filtering logic
- [ ] `ModelService.train_model_task()` - actual training
- [ ] `ModelService.generate_climbs()` - actual generation
- [ ] Integration with existing `climb_mlp_utils.py`

## Integrating Your ML Code

The `ModelService.train_model_task()` method has placeholder comments showing where to integrate your existing `climb_mlp_utils.py` code. The main steps:

1. Copy your ML code to `app/ml/`
2. Import in `model_service.py`
3. Replace placeholder training loop with actual training
4. Replace placeholder generation with actual model inference
