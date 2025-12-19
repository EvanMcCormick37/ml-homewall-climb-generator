"""
Service layer for business logic.
"""
from app.services.wall_service import WallService
from app.services.climb_service import ClimbService
from app.services.model_service import ModelService
from app.services.job_service import JobService

climb_service = ClimbService()
wall_service = WallService()
model_service = ModelService()
job_service = JobService()

__all__ = [
    "wall_service",
    "climb_service",
    "model_service",
    "job_service"
]
