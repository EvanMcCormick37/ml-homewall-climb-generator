"""
Pydantic schemas for background job tracking.
"""
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobType(str, Enum):
    """Types of background jobs."""
    TRAIN_MODEL = "train_model"


class Job(BaseModel):
    """Job status response."""
    id: str
    job_type: JobType
    status: JobStatus
    progress: float = Field(..., ge=0, le=1)
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobNotFoundError(BaseModel):
    """Error response when job not found."""
    detail: str = "Job not found"
