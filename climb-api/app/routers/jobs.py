"""
Router for background job tracking.

Endpoints:
- GET /jobs/{job_id} - Get job status
"""
from fastapi import APIRouter, HTTPException

from app.schemas import Job
from app.services import job_service

router = APIRouter()


@router.get(
    "/{job_id}",
    response_model=Job,
    summary="Get job status",
    description="Get the status of a background job.",
    responses={
        200: {
            "description": "Job status",
            "content": {
                "application/json": {
                    "examples": {
                        "pending": {
                            "summary": "Job pending",
                            "value": {
                                "id": "job-123",
                                "job_type": "train_model",
                                "status": "PENDING",
                                "progress": 0.0,
                                "created_at": "2024-01-15T10:30:00Z",
                            },
                        },
                        "processing": {
                            "summary": "Job in progress",
                            "value": {
                                "id": "job-123",
                                "job_type": "train_model",
                                "status": "PROCESSING",
                                "progress": 0.45,
                                "created_at": "2024-01-15T10:30:00Z",
                                "started_at": "2024-01-15T10:30:01Z",
                            },
                        },
                        "completed": {
                            "summary": "Job completed",
                            "value": {
                                "id": "job-123",
                                "job_type": "train_model",
                                "status": "COMPLETED",
                                "progress": 1.0,
                                "result": {"model_id": "model-456", "val_loss": 0.032},
                                "created_at": "2024-01-15T10:30:00Z",
                                "started_at": "2024-01-15T10:30:01Z",
                                "completed_at": "2024-01-15T10:35:00Z",
                            },
                        },
                        "failed": {
                            "summary": "Job failed",
                            "value": {
                                "id": "job-123",
                                "job_type": "train_model",
                                "status": "FAILED",
                                "progress": 0.5,
                                "error": "Out of memory during training",
                                "created_at": "2024-01-15T10:30:00Z",
                                "started_at": "2024-01-15T10:30:01Z",
                                "completed_at": "2024-01-15T10:32:00Z",
                            },
                        },
                    }
                }
            },
        },
        404: {"description": "Job not found"},
    },
)
def get_job_status(job_id: str):
    """
    Get the current status of a background job.
    
    Poll this endpoint to track progress of long-running tasks like model training.
    
    Status values:
    - PENDING: Job created but not yet started
    - PROCESSING: Job is currently running
    - COMPLETED: Job finished successfully (check `result` field)
    - FAILED: Job failed (check `error` field)
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
