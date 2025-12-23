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
from fastapi import HTTPException
from app.schemas.jobs import JobStatus
from app.services.job_service import JobService
from app.services.climb_service import ClimbService
from app.services.utils import ClimbGenerator, collate_sequences, create_model_instance, extract_hold_features, run_epoch, process_training_data
from app.config import settings


class ModelService:
    """Service for model operations."""
    
    def __init__(self):
        self.job_service = JobService()
        self.climb_service = ClimbService()
    
    def get_models_for_wall(self, wall_id: str) -> list[ModelSummary]:
        """Get all models for a wall."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE wall_id = ?", (wall_id,)
            ).fetchall()
        
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
        )
            for row in rows
        ]
    
    def get_model(self, wall_id: str, model_id: str) -> ModelSummary | None:
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
    
    def create_model(self, wall_id: str, config: ModelCreate) -> str:
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
    
    def delete_model(self, wall_id: str, model_id: str) -> bool:
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
    
    def train_model_task(
        self,
        job_id: str,
        model_id: str,
        wall_id: str,
        config: ModelCreate,
    ):
        """
        Background task for training a model.
        
        This is called via BackgroundTasks.add_task() and runs
        in the same process as the API server.
        
        Args:
            job_id: Job ID for progress tracking
            model_id: Model to train
            wall_id: Wall the model belongs to
            config: Training configuration
        """
        try:
            # Update job status to processing
            self.job_service.update_job_status(
                job_id, JobStatus.PROCESSING, progress=0.0
            )
            
            # Update model status
            self._update_model_status(model_id, ModelStatus.TRAINING)
            # Return the nn.Module instance of a model, and a Boolean indicating whether it's sequential
            model = create_model_instance(config.model_type, config.features)
            
            # Preprocess the training data and hold-id<->feature map.
            train_ds, val_ds, hold_map, num_climbs, num_moves = process_training_data(
                wall_id,
                feature_config=config.features,
                sequential=model.is_sequential,
                augment=config.augment_dataset,
                val_split=config.val_split,
            )
            
            # Collate sequential training data
            train_loader = DataLoader(train_ds, batch_size=settings.BATCH_SIZE, shuffle=True, collate_fn=(collate_sequences if model.is_sequential else None))
            val_loader = DataLoader(val_ds, batch_size=settings.BATCH_SIZE, shuffle=False, collate_fn=(collate_sequences if model.is_sequential else None))

            save_weights_path = settings.WALLS_DIR / wall_id / model_id / "best.pth"
            optimizer = optim.Adam(model.parameters(),lr=settings.LR)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')

            for epoch in range(config.epochs):
                run_epoch(model, model.is_sequential, train_loader, criterion, optimizer, settings.DEVICE)
                val_loss, _ = run_epoch(model, model.is_sequential, val_loader, criterion, None, settings.DEVICE)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_weights_path)
                
                if epoch > 0 and (epoch % max(1,config.epochs//10) == 0):
                    progress = float(epoch/config.epochs)
                    self.job_service.update_job_status(job_id, JobStatus.PROCESSING, progress=progress)
            
            self._complete_model_training(model_id, val_loss, config.epochs, num_climbs, num_moves)
            
            # Complete job
            self.job_service.complete_job(
                job_id, 
                result={"model_id": model_id, "val_loss": val_loss}
            )
            
        except Exception as e:
            # Update model status to failed
            self._update_model_status(model_id, ModelStatus.FAILED)
            
            # Fail job with error message
            self.job_service.fail_job(job_id, str(e))
            
            # Log full traceback for debugging
            print(f"Training failed: {traceback.format_exc()}")
    
    def generate_climbs(
        self,
        wall_id: str,
        model_id: str,
        request: GenerateRequest,
    ) -> list[GeneratedClimb]:
        """
        Generate climbs using a trained model.
        
        Args:
            wall_id: The wall ID
            model_id: The model ID
            request: Generation parameters
            
        Returns:
            List of generated climbs
        """

        # 1. Get model parameters from models table and use them to instantiate model
        with get_db() as conn:
            query = f"SELECT model_type, features FROM models WHERE id LIKE ?"
            params = [model_id]
            row = conn.execute(query, params).fetchone()
            model_type = row["model_type"]
            features = json.loads(row["features"])
        feature_config = FeatureConfig.model_validate(features)

        model = create_model_instance(model_type, feature_config)
        
        # 2. Rehydrate the model from the features and model path.
        model_weights_path = settings.WALLS_DIR / wall_id / model_id / "best.pth"
        model.load_state_dict(torch.load(model_weights_path))

        # 3. Load hold map and generator and generate climbs
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
    
    def _update_model_status(self, model_id: str, status: ModelStatus):
        """Update a model's status."""
        with get_db() as conn:
            conn.execute(
                "UPDATE models SET status = ? WHERE id = ?",
                (status.value, model_id),
            )
    
    def _complete_model_training(
        self, 
        model_id: str,
        val_loss: float,
        epochs: int,
        climbs_trained: int,
        moves_trained: int,
    ):
        """Mark model as successfully trained."""
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
