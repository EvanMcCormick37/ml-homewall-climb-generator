"""
Service for managing ML models.

Handles:
- Model CRUD operations
- Background training tasks
- Climb generation
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
import traceback

from app.database import get_db, WALLS_DIR
from app.schemas import (
    ModelCreate,
    ModelDetail,
    ModelSummary,
    ModelStatus,
    FeatureConfig,
    GenerateRequest,
    GeneratedClimb,
)
from app.schemas.jobs import JobStatus
from app.services.job_service import JobService
from app.services.climb_service import ClimbService


class ModelService:
    """Service for model operations."""
    
    def __init__(self):
        self.job_service = JobService()
        self.climb_service = ClimbService()
    
    def _get_models_dir(self, wall_id: str) -> Path:
        """Get directory for a wall's models."""
        return WALLS_DIR / wall_id / "models"
    
    def _get_model_path(self, wall_id: str, model_id: str) -> Path:
        """Get path to a model's weights file."""
        return self._get_models_dir(wall_id) / f"{model_id}.pth"
    
    def get_models_for_wall(self, wall_id: str) -> list[ModelSummary]:
        """Get all models for a wall."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE wall_id = ?", (wall_id,)
            ).fetchall()
        
        return [
            ModelSummary(
                id=row["id"],
                model_type=row["model_type"],
                status=ModelStatus(row["status"]),
                moves_trained=row["moves_trained"],
                climbs_trained=row["climbs_trained"],
                val_loss=row["val_loss"],
                created_at=row["created_at"],
                trained_at=row["trained_at"],
            )
            for row in rows
        ]
    
    def get_model(self, wall_id: str, model_id: str) -> ModelDetail | None:
        """Get detailed model info."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE id = ? AND wall_id = ?",
                (model_id, wall_id),
            ).fetchone()
        
        if not row:
            return None
        
        return ModelDetail(
            id=row["id"],
            wall_id=row["wall_id"],
            model_type=row["model_type"],
            features=FeatureConfig(**json.loads(row["features"])),
            status=ModelStatus(row["status"]),
            val_loss=row["val_loss"],
            epochs_trained=row["epochs_trained"],
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
        self._get_models_dir(wall_id).mkdir(parents=True, exist_ok=True)
        
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
        model_path = self._get_model_path(wall_id, model_id)
        if model_path.exists():
            model_path.unlink()
        
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
            
            # ===== TRAINING LOGIC GOES HERE =====
            # TODO: Implement actual training
            #
            # 1. Load wall data (holds)
            # wall_json_path = WALLS_DIR / wall_id / "wall.json"
            # with open(wall_json_path) as f:
            #     wall_data = json.load(f)
            # hold_map = {h["hold_id"]: h for h in wall_data["holds"]}
            #
            # 2. Load climb sequences
            # climbs = self.climb_service.get_climbs_for_training(wall_id)
            #
            # 3. Preprocess into training format
            # (Use your existing climb_mlp_utils functions)
            #
            # 4. Train model with progress updates
            # for epoch in range(config.epochs):
            #     # ... training step ...
            #     progress = (epoch + 1) / config.epochs
            #     self.job_service.update_job_status(job_id, JobStatus.PROCESSING, progress)
            #
            # 5. Save trained weights
            # model_path = self._get_model_path(wall_id, model_id)
            # torch.save(model.state_dict(), model_path)
            #
            # 6. Record final validation loss
            # val_loss = ...
            #
            # ====================================
            
            # Placeholder: simulate training
            import time
            for i in range(10):
                time.sleep(0.5)  # Simulate work
                progress = (i + 1) / 10
                self.job_service.update_job_status(
                    job_id, JobStatus.PROCESSING, progress=progress
                )
            
            val_loss = 0.032  # Placeholder
            
            # Update model as trained
            self._complete_model_training(model_id, val_loss, config.epochs)
            
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
        # TODO: Implement actual generation
        #
        # 1. Load model weights
        # model_path = self._get_model_path(wall_id, model_id)
        # model = ClimbMLP()  # or ClimbLSTM based on model_type
        # model.load_state_dict(torch.load(model_path))
        #
        # 2. Load hold map
        # wall_json_path = WALLS_DIR / wall_id / "wall.json"
        # with open(wall_json_path) as f:
        #     wall_data = json.load(f)
        # hold_map = {h["hold_id"]: extract_hold_features(h) for h in wall_data["holds"]}
        #
        # 3. Create generator
        # generator = ClimbGenerator(model, hold_map, device="cpu")
        #
        # 4. Generate climbs
        # climbs = []
        # for _ in range(request.num_climbs):
        #     sequence = generator.generate(
        #         start_lh=request.starting_holds[0],
        #         start_rh=request.starting_holds[1],
        #         max_moves=request.max_moves,
        #         temperature=request.temperature,
        #     )
        #     climbs.append(GeneratedClimb(sequence=sequence, num_moves=len(sequence)))
        #
        # return climbs
        
        # Placeholder: return dummy data
        return [
            GeneratedClimb(
                sequence=[[request.starting_holds[0], request.starting_holds[1]]] 
                         + [[i, i+1] for i in range(request.max_moves)],
                num_moves=request.max_moves + 1,
            )
            for _ in range(request.num_climbs)
        ]
    
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
        epochs: int
    ):
        """Mark model as successfully trained."""
        with get_db() as conn:
            conn.execute(
                """
                UPDATE models 
                SET status = ?, val_loss = ?, epochs_trained = ?, trained_at = ?
                WHERE id = ?
                """,
                (
                    ModelStatus.TRAINED.value,
                    val_loss,
                    epochs,
                    datetime.utcnow().isoformat(),
                    model_id,
                ),
            )
