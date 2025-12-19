"""
Climb Generator API - Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import walls, climbs, models, jobs
from app.database import init_db

app = FastAPI(
    title="Climb Generator API",
    description="API for managing climbing walls, climbs, and ML-based climb generation",
    version="0.1.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(walls.router, prefix="/walls", tags=["walls"])
app.include_router(climbs.router, prefix="/walls/{wall_id}/climbs", tags=["climbs"])
app.include_router(models.router, prefix="/walls/{wall_id}/models", tags=["models"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    init_db()


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
