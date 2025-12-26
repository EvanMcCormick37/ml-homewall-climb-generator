## Claude Opus 4.5 Overview
# Climb Generator API

A REST API for managing climbing walls, climbs, and ML-based climb generation using PyTorch models (MLP, RNN, LSTM).

## Project Structure

```
climb-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Application settings and configuration
│   ├── database.py          # SQLite setup and connection management
│   ├── routers/             # API endpoint definitions
│   │   ├── walls.py         # Wall CRUD endpoints
│   │   ├── climbs.py        # Climb CRUD endpoints
│   │   ├── models.py        # Model training/generation endpoints
│   │   └── jobs.py          # Background job tracking
│   ├── schemas/             # Pydantic request/response models
│   │   ├── base.py          # Shared types (HoldPosition, HoldDetail)
│   │   ├── walls.py         # Wall and hold schemas
│   │   ├── climbs.py        # Climb schemas with filtering/sorting
│   │   ├── models.py        # Model configs, generation requests
│   │   └── jobs.py          # Job status tracking
│   ├── services/            # Business logic layer
│   │   ├── container.py         # Service function container
│   │   ├── wall_service.py      # Wall CRUD, photo handling
│   │   ├── climb_service.py     # Climb CRUD with advanced filtering
│   │   ├── model_service.py     # Model training & generation
│   │   ├── job_service.py       # Background job queue
│   │   └── utils/               # Model training utilities
│   │       ├── model_utils.py       # PyTorch models (MLP, RNN, LSTM)
│   │       └── train_data_utils.py  # Data preprocessing, augmentation
│   └── test/
│       └── test_api.py      # API endpoint tests
├── data/                    # Created at runtime
│   ├── storage.db           # SQLite database
│   └── walls/               # Wall data directories
│       └── {wall-id}/
│           ├── holds.json   # Hold definitions with features
│           ├── photo.jpg    # Wall photo
│           └── models/      # Trained model weights (.pth files)
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
- Health Check: http://localhost:8000/health

## API Endpoints

### Walls

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/walls` | List all walls |
| POST | `/api/v1/walls` | Create a new wall |
| GET | `/api/v1/walls/{wall_id}` | Get wall details with holds |
| DELETE | `/api/v1/walls/{wall_id}` | Delete wall and all data |
| PUT | `/api/v1/walls/{wall_id}/holds` | Add or replace wall holdset |
| GET | `/api/v1/walls/{wall_id}/photo` | Get wall photo |
| PUT | `/api/v1/walls/{wall_id}/photo` | Upload/replace wall photo |

**Create Wall (multipart/form-data):**
- `name` (required): Wall name (1-100 characters)
- `photo` (required): Wall photo (JPEG or PNG)
- `dimensions` (optional): Comma-separated "width,height" in cm
- `angle` (optional): Wall angle in degrees from vertical

### Climbs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/walls/{wall_id}/climbs` | List climbs with filters |
| POST | `/api/v1/walls/{wall_id}/climbs` | Create a climb |
| DELETE | `/api/v1/walls/{wall_id}/climbs/{climb_id}` | Delete a climb |

**Climb Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grade_range` | int[] | [0, 180] | Min/max V-grade range |
| `include_projects` | bool | true | Include ungraded climbs |
| `setter` | string | - | Filter by setter ID |
| `name_includes` | string | - | Filter by name (partial match) |
| `holds_include` | int[] | - | Hold IDs that must ALL be in climb |
| `tags_include` | string[] | - | Tags that must ALL be present |
| `after` | datetime | - | Climbs created after this date |
| `sort_by` | enum | date | Sort by: date, name, grade, ticks, num_moves |
| `descending` | bool | true | Sort descending |
| `limit` | int | 50 | Max results (1-200) |
| `offset` | int | 0 | Pagination offset |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/walls/{wall_id}/models` | List models |
| POST | `/api/v1/walls/{wall_id}/models` | Create and train model |
| GET | `/api/v1/walls/{wall_id}/models/{model_id}` | Get model details |
| DELETE | `/api/v1/walls/{wall_id}/models/{model_id}` | Delete model |
| POST | `/api/v1/walls/{wall_id}/models/{model_id}/generate` | Generate climbs |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/jobs/{job_id}` | Get background job status |

**Job Status Values:** `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`

## Data Model

### Hold Features
Each hold has the following properties:
| Property | Type | Description |
|----------|------|-------------|
| `hold_id` | int | Unique identifier |
| `norm_x`, `norm_y` | float | Normalized position (0-1) |
| `pull_x`, `pull_y` | float | Pull direction vector (-1 to 1) |
| `useability` | float | Difficulty rating (0-10) |

### Climb Sequence
Climbs are stored as sequences of hand positions:
```json
[[lh_hold_id, rh_hold_id], [lh_hold_id, rh_hold_id], ...]
```
Where `-1` indicates a limb is off the wall.

### Model Architecture Options
| Type | Description |
|------|-------------|
| `mlp` | Simple feedforward network for next-move prediction |
| `rnn` | Vanilla RNN with hidden state for sequence modeling |
| `lstm` | LSTM with cell state for better long-term dependencies |

### Feature Configuration
Models can be trained with different feature subsets:
```json
{
  "position": true,        // Use hold x,y coordinates
  "pull_direction": true,  // Use pull vector
  "difficulty": true       // Use useability rating
}
```

## Training Pipeline

1. **Data Loading**: Climbs fetched from database for specified wall
2. **Feature Extraction**: Holds mapped to feature vectors based on config
3. **Augmentation** (optional): 6x expansion via mirroring and translation
4. **Train/Val Split**: 80/20 split before augmentation
5. **Dataset Creation**: 
   - `ClimbDataset` for MLP (position pairs)
   - `ClimbSequenceDataset` for RNN/LSTM (full sequences)
6. **Training Loop**:
   - Adam optimizer (lr=0.001)
   - MSE loss
   - Validation tracking
   - Model checkpointing (saves best val_loss)
7. **Job Progress**: Updates tracked in jobs table

**Training Stats Tracked:**
- `val_loss`: Best validation loss
- `epochs_trained`: Number of epochs completed
- `climbs_trained`: Number of unique climbs in training set
- `moves_trained`: Total move count in training data

## Configuration

Settings are managed in `app/config.py` with environment variable support:

```python
# Model defaults
NUM_LIMBS = 2           # Left hand, right hand
HIDDEN_DIM = 256        # Hidden layer size
N_HIDDEN_LAYERS = 3     # Number of hidden layers

# Training defaults
VAL_SPLIT = 0.2         # 20% validation set
EPOCHS = 100            # Default training epochs
MAX_EPOCHS = 1000       # Maximum allowed epochs
LR = 0.001              # Learning rate
DEVICE = "cpu"          # Training device
AUGMENT_DATA = True     # Enable data augmentation
```

## Example Usage

### Create a Wall

```bash
curl -X POST http://localhost:8000/api/v1/walls \
  -F "name=Home Wall" \
  -F "photo=@wall_photo.jpg" \
  -F "dimensions=244,300" \
  -F "angle=15"
```

### Add Holds to a Wall

```bash
curl -X PUT http://localhost:8000/api/v1/walls/{wall_id}/holds \
  -F 'holds=[{"hold_id":0,"norm_x":0.2,"norm_y":0.1,"pull_x":0,"pull_y":1,"useability":8}]'
```

### Create a Climb

```bash
curl -X POST http://localhost:8000/api/v1/walls/{wall_id}/climbs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First V3",
    "grade": 30,
    "setter": "user123",
    "sequence": [[1, 2], [3, 2], [3, 4], [5, 4]],
    "tags": ["dynamic", "overhang"]
  }'
```

### List Climbs with Filters

```bash
# Get all V4-V6 climbs by a specific setter, sorted by date
curl "http://localhost:8000/api/v1/walls/{wall_id}/climbs?\
grade_range=40,60&\
setter=user123&\
sort_by=date&\
descending=true&\
limit=20"
```

### Train a Model

```bash
# Start training (returns job_id for polling)
curl -X POST http://localhost:8000/api/v1/walls/{wall_id}/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "lstm",
    "features": {
      "position": true,
      "pull_direction": true,
      "difficulty": true
    },
    "epochs": 200,
    "augment_dataset": true
  }'
# Returns: {"model_id": "...", "job_id": "..."}
```

### Poll Training Progress

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
# Returns: {"status": "PROCESSING", "progress": 0.45, ...}
```

### Generate Climbs

```bash
curl -X POST http://localhost:8000/api/v1/walls/{wall_id}/models/{model_id}/generate \
  -H "Content-Type: application/json" \
  -d '{
    "starting_holds": [0, 1],
    "max_moves": 10,
    "num_climbs": 5,
    "temperature": 0.01,
    "force_alternating": true
  }'
```

## Architecture

### Service Layer Design
The application uses a dependency injection pattern via `ServiceContainer`:

```python
services = ServiceContainer(
    create_job=job_service.create_job,
    get_climbs=climb_service.get_climbs,
    train_model_task=model_service.make_train_model_task(...),
    # ... all service functions wired up
)
```

This allows easy testing and swapping of implementations.

### Why SQLite?
- Simple setup for prototype/development
- Sufficient for single-user or small-scale deployments
- Easy to backup (single file)
- Can be migrated to PostgreSQL if needed

### Why JSON Files for Hold Data?
- Hold data is read-heavy, rarely updated
- Easier to visualize and manually edit if needed
- Keeps database queries fast (only metadata in DB)

### Background Job Queue
- SQLite-based queue for background tasks
- Runs in same process (no external workers needed)
- Progress tracking via periodic status updates
- For production, consider Celery + Redis

## Testing

```bash
# Run all tests
pytest app/test/test_api.py -v

# Run specific test class
pytest app/test/test_api.py::TestClimbEndpoints -v
```

## Development Status

### Implemented Features
- ✅ Wall CRUD with photo management
- ✅ Hold management (set/replace holdsets)
- ✅ Climb CRUD with advanced filtering and sorting
- ✅ Model training with MLP, RNN, LSTM architectures
- ✅ Background job system with progress tracking
- ✅ Data augmentation (mirroring, translation)
- ✅ Model checkpointing and validation tracking

### Planned Enhancements
- [ ] Implement full autoregressive climb generation
- [ ] Add model evaluation metrics beyond val_loss
- [ ] Implement climb difficulty prediction
- [ ] Add support for foothold tracking (currently only tracks hands)
- [ ] Add authentication and user management
- [ ] Add rate limiting for API endpoints
- [ ] Add model versioning and A/B testing support

## License

MIT License
