"""
Service for managing ML models.

Handles:
- Model CRUD operations
- Background training tasks
- Climb generation
"""
import json
import uuid
import shutil
from datetime import datetime
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.database import get_db
from app.schemas import (
    ModelCreate,
    ModelSummary,
    ModelStatus,
    FeatureConfig,
    GenerateRequest,
    GeneratedClimb,
)
from app.schemas.jobs import JobStatus
from app.services.utils import ClimbGenerator, collate_sequences, create_model_instance, extract_hold_features, run_epoch, process_training_data
from app.config import settings

from typing import Callable, Any

# Type aliases for the JobService dependencies
UpdateJobStatus = Callable[[str, JobStatus, float], None]
CompleteJob = Callable[[str, dict[str, Any]], None]
FailJob = Callable[[str, str], None]

def update_model_status(model_id: str, status: ModelStatus):
    """Update a model's status in the DB."""
    with get_db() as conn:
        conn.execute(
            "UPDATE models SET status = ? WHERE id = ?",
            (status.value, model_id),
        )

def complete_model_training(
    model_id: str,
    val_loss: float,
    epochs: int,
    climbs_trained: int,
    moves_trained: int,
):
    """Mark model as successfully trained in the DB."""
    with get_db() as conn:
        conn.execute(
            """
            UPDATE models 
            SET status = ?, val_loss = ?, epochs_trained = ?, climbs_trained = ?, moves_trained = ?, trained_at = ?
            WHERE id = ?
            """,
            (
                ModelStatus.TRAINED.value,
                val_loss,
                epochs,
                climbs_trained,
                moves_trained,
                datetime.now(),
                model_id,
            ),
        )

def get_models_for_wall(wall_id: str) -> list[ModelSummary]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM models WHERE wall_id = ?", (wall_id,)).fetchall()
    
    return [
        ModelSummary(
            id=row["id"],
            wall_id=row["wall_id"],
            model_type=row["model_type"],
            features=FeatureConfig(**json.loads(row["features"])),
            status=ModelStatus(row["status"]),
            val_loss=row["val_loss"],
            epochs_trained=row["epochs_trained"],
            climbs_trained=row["climbs_trained"],
            moves_trained=row["moves_trained"],
            created_at=row["created_at"],
            trained_at=row["trained_at"],
        ) for row in rows
    ]

def generate_climbs(
    wall_id: str,
    model_id: str,
    request: GenerateRequest,
) -> list[GeneratedClimb]:
    # 1. Get model params
    with get_db() as conn:
        row = conn.execute("SELECT model_type, features FROM models WHERE id = ?", (model_id,)).fetchone()
    
    feature_config = FeatureConfig.model_validate(json.loads(row["features"]))
    model = create_model_instance(row["model_type"], feature_config)
    
    # 2. Load weights
    model_weights_path = settings.WALLS_DIR / wall_id / model_id / "best.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))

    # 3. Generate
    wall_json_path = settings.WALLS_DIR / wall_id / "wall.json"
    with open(wall_json_path) as f:
        wall_data = json.load(f)
        
    hold_map = {h["hold_id"]: extract_hold_features(h, feature_config) for h in wall_data["holds"]}
    generator = ClimbGenerator(model, hold_map, device="cpu")
    
    climbs = []
    for _ in range(request.num_climbs):
        sequence = generator.generate(
            start_lh=request.starting_holds[0],
            start_rh=request.starting_holds[1],
            max_moves=request.max_moves,
            temperature=request.temperature,
            force_alternating=request.force_alternating,
        )
        climbs.append(GeneratedClimb(sequence=sequence, num_moves=len(sequence)))
    
    return climbs

def get_model(wall_id: str, model_id: str) -> ModelSummary | None:
    """Get detailed model info."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE id = ? AND wall_id = ?",
            (model_id, wall_id),
        ).fetchone()
    
    if not row:
        return None
    
    return ModelSummary(
        id=row["id"],
        wall_id=row["wall_id"],
        model_type=row["model_type"],
        features=FeatureConfig(**json.loads(row["features"])),
        status=ModelStatus(row["status"]),
        val_loss=row["val_loss"],
        epochs_trained=row["epochs_trained"],
        climbs_trained=row["climbs_trained"],
        moves_trained=row["moves_trained"],
        created_at=row["created_at"],
        trained_at=row["trained_at"],
    )

def create_model(wall_id: str, config: ModelCreate) -> str:
    """
    Create a new model record (does not start training).
    
    Args:
        wall_id: The wall ID
        config: Model configuration
        
    Returns:
        The new model ID
    """
    model_id = f"model-{uuid.uuid4().hex[:12]}"
    
    # Ensure models directory exists
    model_dir = settings.WALLS_DIR / wall_id / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO models (id, wall_id, model_type, features, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                model_id,
                wall_id,
                config.model_type.value,
                json.dumps(config.features.model_dump()),
                ModelStatus.UNTRAINED.value,
            ),
        )
    
    return model_id

def delete_model(wall_id: str, model_id: str) -> bool:
    """Delete a model and its weights file."""
    # Delete from database
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM models WHERE id = ? AND wall_id = ?",
            (model_id, wall_id),
        )
    
    if cursor.rowcount == 0:
        return False
    
    # Delete weights file if exists
    model_dir = settings.WALLS_DIR / wall_id / model_id
    if model_dir.exists():
        shutil.rmtree(model_dir)
    
    return True

def make_train_model_task(
    on_update: UpdateJobStatus,  # Injected
    on_complete: CompleteJob,    # Injected
    on_fail: FailJob,            # Injected
):
    def train_model_task(
        job_id: str,
        model_id: str,
        wall_id: str,
        config: ModelCreate,
    ):
        try:
            on_update(job_id, JobStatus.PROCESSING, 0.0)
            update_model_status(model_id, ModelStatus.TRAINING)
            
            model = create_model_instance(config.model_type, config.features)
            
            train_ds, val_ds, hold_map, num_climbs, num_moves = process_training_data(
                wall_id,
                feature_config=config.features,
                sequential=model.is_sequential,
                augment=config.augment_dataset,
                val_split=config.val_split,
            )
            
            train_loader = DataLoader(train_ds, batch_size=settings.BATCH_SIZE, shuffle=True, 
                                    collate_fn=(collate_sequences if model.is_sequential else None))
            val_loader = DataLoader(val_ds, batch_size=settings.BATCH_SIZE, shuffle=False, 
                                    collate_fn=(collate_sequences if model.is_sequential else None))

            save_weights_path = settings.WALLS_DIR / wall_id / model_id / "best.pth"
            optimizer = optim.Adam(model.parameters(), lr=settings.LR)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')

            for epoch in range(config.epochs):
                run_epoch(model, model.is_sequential, train_loader, criterion, optimizer, settings.DEVICE)
                val_loss, _ = run_epoch(model, model.is_sequential, val_loader, criterion, None, settings.DEVICE)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_weights_path)
                
                if epoch > 0 and (epoch % max(1, config.epochs // 10) == 0):
                    on_update(job_id, JobStatus.PROCESSING, float(epoch / config.epochs))
            
            complete_model_training(model_id, best_val_loss, config.epochs, num_climbs, num_moves)
            on_complete(job_id, {"model_id": model_id, "val_loss": best_val_loss})
                
        except Exception as e:
            update_model_status(model_id, ModelStatus.FAILED)
            on_fail(job_id, str(e))
            print(f"Training failed: {traceback.format_exc()}")
    
    return train_model_task
