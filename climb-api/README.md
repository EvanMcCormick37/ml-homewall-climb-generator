# Claude Sonnet 4.5 overview

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
│   │   ├── walls.py         # Wall CRUD endpoints (needs router connection)
│   │   ├── climbs.py        # Climb CRUD endpoints (✓ functional)
│   │   ├── models.py        # Model training/generation endpoints (needs router connection)
│   │   └── jobs.py          # Background job tracking (needs router connection)
│   ├── schemas/             # Pydantic request/response models
│   │   ├── walls.py         # Wall and hold schemas
│   │   ├── climbs.py        # Climb schemas with filtering/sorting
│   │   ├── models.py        # Model configs, generation requests
│   │   └── jobs.py          # Job status tracking
│   └── services/            # Business logic (✓ fully implemented)
│       ├── wall_service.py      # Wall CRUD, photo handling
│       ├── climb_service.py     # Climb CRUD with advanced filtering
│       ├── model_service.py     # Model training & generation (✓ training complete)
│       ├── job_service.py       # Background job queue
│       └── utils/               # ML utilities
│           ├── model_utils.py       # PyTorch models (MLP, RNN, LSTM)
│           └── train_data_utils.py  # Data preprocessing, augmentation
├── data/                    # Created at runtime
│   ├── storage.db          # SQLite database
│   └── walls/              # Wall data directories
│       └── {wall-id}/
│           ├── wall.json   # Hold definitions with features
│           ├── photo.jpg   # Wall photo
│           └── models/     # Trained model weights (.pth files)
├── requirements.txt
└── README.md
```

## Features

### ✅ Fully Implemented
- **Database Schema**: SQLite with tables for walls, climbs, models, and jobs
- **Climb Management**: Full CRUD with advanced filtering (by grade, setter, holds, tags, date)
- **Service Layer**: Complete business logic for all entities
- **ML Training Pipeline**: 
  - Three model architectures (MLP, RNN, LSTM)
  - Feature extraction (position, pull direction, difficulty)
  - Data augmentation (mirroring, translation, 6x expansion)
  - Sequential and non-sequential training loops
  - Validation tracking and model checkpointing
- **Background Job System**: SQLite-based job queue with status tracking

### 🚧 Partially Implemented
- **Climb Endpoints**: Functional but DELETE has a minor bug (returns 501 after successful deletion)
- **Model Generation**: Service returns placeholder data; needs full autoregressive generation

### ⏳ Needs Router Connection
The following services are fully implemented but need to be connected in their routers:
- **Wall Endpoints**: All return 501, need to call wall_service methods
- **Model Endpoints**: All return 501, need to call model_service methods
- **Job Endpoints**: Returns 501, need to call job_service methods

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

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/walls` | List all walls | ⏳ Service ready |
| POST | `/walls` | Create a new wall | ⏳ Service ready |
| GET | `/walls/{wall_id}` | Get wall details with holds | ⏳ Service ready |
| DELETE | `/walls/{wall_id}` | Delete wall and all data | ⏳ Service ready |
| GET | `/walls/{wall_id}/photo` | Get wall photo | ⏳ Service ready |
| PUT | `/walls/{wall_id}/photo` | Upload wall photo | ⏳ Service ready |

### Climbs

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/walls/{wall_id}/climbs` | List climbs (with filters) | ✅ Working |
| POST | `/walls/{wall_id}/climbs` | Create a climb | ✅ Working |
| DELETE | `/walls/{wall_id}/climbs/{climb_id}` | Delete a climb | 🐛 Minor bug |

**Climb Query Parameters:**
- `grade_range` - Filter by V-grade range (e.g., "0,180" for all grades)
- `include_projects` - Include ungraded climbs (default: true)
- `setter` - Filter by setter ID
- `name_includes` - Filter by name (partial match)
- `holds_include` - Comma-separated hold IDs that must ALL be in the climb
- `tags_include` - Comma-separated tags that must ALL be in the climb
- `after` - Climbs created after this datetime
- `sort_by` - Sort by: date, name, grade, ticks, num_moves
- `descending` - Sort descending (default: true)
- `limit` / `offset` - Pagination (default limit: 50)

### Models

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/walls/{wall_id}/models` | List models | ⏳ Service ready |
| POST | `/walls/{wall_id}/models` | Create and train model | ⏳ Service ready |
| GET | `/walls/{wall_id}/models/{model_id}` | Get model details | ⏳ Service ready |
| DELETE | `/walls/{wall_id}/models/{model_id}` | Delete model | ⏳ Service ready |
| POST | `/walls/{wall_id}/models/{model_id}/generate` | Generate climbs | 🚧 Returns placeholders |

### Jobs

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/jobs/{job_id}` | Get background job status | ⏳ Service ready |

## Data Model

### Hold Features
Each hold has the following features:
- `hold_id`: Unique identifier (int)
- `norm_x`, `norm_y`: Normalized position (0-1)
- `pull_x`, `pull_y`: Pull direction vector (-1 to 1)
- `useability`: Difficulty rating (0-10)

### Climb Sequence
Climbs are stored as sequences of positions:
```json
[[lh_hold_id, rh_hold_id], [lh_hold_id, rh_hold_id], ...]
```
Where `-1` indicates a limb is off the wall.

### Model Architecture Options
- **MLP**: Simple feedforward network for next-move prediction
- **RNN**: Vanilla RNN with hidden state for sequence modeling
- **LSTM**: LSTM with cell state for better long-term dependencies

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

The training system is fully implemented:

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

### Working: Add a Climb

```bash
curl -X POST http://localhost:8000/walls/{wall_id}/climbs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First V3",
    "grade": 30,
    "setter": "user123",
    "sequence": [[1, 2], [3, 2], [3, 4], [5, 4]],
    "tags": ["dynamic", "overhang"]
  }'
```

### Working: List Climbs with Filters

```bash
# Get all V4-V6 climbs by a specific setter, sorted by date
curl "http://localhost:8000/walls/{wall_id}/climbs?\
grade_range=40,60&\
setter=user123&\
sort_by=date&\
descending=true&\
limit=20"
```

### Needs Connection: Train a Model

Once routers are connected, this will work:

```bash
# Start training
curl -X POST http://localhost:8000/walls/{wall_id}/models \
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

# Poll for training progress
curl http://localhost:8000/jobs/{job_id}
# Returns: {"status": "PROCESSING", "progress": 0.45, ...}
```

## Development Status & Next Steps

### Immediate Priorities
1. **Fix Climb Delete Bug**: Remove the `raise HTTPException(status_code=501)` after successful deletion in `routers/climbs.py`
2. **Connect Wall Router**: Uncomment service calls in `routers/walls.py`
3. **Connect Model Router**: Uncomment service calls in `routers/models.py`
4. **Connect Jobs Router**: Uncomment service call in `routers/jobs.py`

### Feature Enhancements
- [ ] Implement full autoregressive climb generation in `ModelService.generate_climbs()`
- [ ] Add model evaluation metrics beyond val_loss
- [ ] Implement climb difficulty prediction
- [ ] Add support for foothold tracking (currently only tracks hands)
- [ ] Add authentication and user management
- [ ] Add rate limiting for API endpoints
- [ ] Add model versioning and A/B testing support

### Known Issues
- Climb DELETE endpoint returns 501 after successful deletion
- Wall router endpoints all return 501 (service is ready)
- Model router endpoints all return 501 (service is ready)  
- Job router endpoint returns 501 (service is ready)
- Generation returns placeholder sequences instead of model predictions

## Architecture Notes

### Why SQLite?
- Simple setup for prototype/development
- Sufficient for single-user or small-scale deployments
- Easy to backup (single file)
- Can be migrated to PostgreSQL if needed

### Why JSON Files for Hold Data?
- Hold data is read-heavy, rarely updated
- Easier to visualize and manually edit if needed
- Keeps database queries fast (only metadata in DB)
- Wall JSON includes full schema for external tools

### Job Queue Design
- Simple SQLite-based queue for background tasks
- Runs in same process (no external workers needed)
- Progress tracking via periodic status updates
- For production, consider Celery + Redis

## License

[Add your license here]
