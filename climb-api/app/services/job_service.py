"""
Service for managing background jobs.

This implements a simple SQLite-based job queue for tracking
the status of long-running tasks like model training.
"""
import json
import uuid
from datetime import datetime
from typing import Optional, Any

from app.database import get_db
from app.schemas.jobs import Job, JobStatus, JobType


class JobService:
    """Service for job queue operations."""
    
    def create_job(
        self, 
        job_type: JobType, 
        params: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Create a new job record.
        
        Args:
            job_type: Type of job (e.g., train_model)
            params: JSON-serializable parameters for the job
            
        Returns:
            The new job ID
        """
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, job_type, status, params)
                VALUES (?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type.value,
                    JobStatus.PENDING.value,
                    json.dumps(params) if params else None,
                ),
            )
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job status by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            Job object or None if not found
        """
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            
        if not row:
            return None
            
        return Job(
            id=row["id"],
            job_type=JobType(row["job_type"]),
            status=JobStatus(row["status"]),
            progress=row["progress"],
            error=row["error"],
            result=json.loads(row["result"]) if row["result"] else None,
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
    ):
        """
        Update job status and progress.
        
        Args:
            job_id: The job ID
            status: New status
            progress: Progress value (0.0 to 1.0)
        """
        updates = ["status = ?"]
        values = [status.value]
        
        if progress is not None:
            updates.append("progress = ?")
            values.append(progress)
            
        if status == JobStatus.PROCESSING:
            updates.append("started_at = ?")
            values.append(datetime.utcnow().isoformat())
            
        values.append(job_id)
        
        with get_db() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                values,
            )
    
    def complete_job(
        self,
        job_id: str,
        result: Optional[dict[str, Any]] = None,
    ):
        """
        Mark a job as completed.
        
        Args:
            job_id: The job ID
            result: JSON-serializable result data
        """
        with get_db() as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, progress = 1.0, result = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.COMPLETED.value,
                    json.dumps(result) if result else None,
                    datetime.utcnow().isoformat(),
                    job_id,
                ),
            )
    
    def fail_job(self, job_id: str, error: str):
        """
        Mark a job as failed.
        
        Args:
            job_id: The job ID
            error: Error message
        """
        with get_db() as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, error = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.FAILED.value,
                    error,
                    datetime.utcnow().isoformat(),
                    job_id,
                ),
            )
