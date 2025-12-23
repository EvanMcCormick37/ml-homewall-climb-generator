"""
Service functions for managing background jobs.
"""
import json
import uuid
from datetime import datetime
from typing import Any

from app.database import get_db
from app.schemas.jobs import Job, JobStatus, JobType


def create_job(
    job_type: JobType, 
    params: dict[str, Any] | None = None
) -> str:
    """Create a new job record."""
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, job_type, status, params)
            VALUES (?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                JobStatus.PENDING,
                json.dumps(params) if params else None,
            ),
        )
    
    return job_id


def get_job(job_id: str) -> Job | None:
    """Get job status by ID."""
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
    job_id: str,
    status: JobStatus,
    progress: float | None = None,
):
    """Update job status and progress."""
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
    job_id: str,
    result: dict[str, Any] | None = None,
):
    """Mark a job as completed."""
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


def fail_job(job_id: str, error: str):
    """Mark a job as failed."""
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