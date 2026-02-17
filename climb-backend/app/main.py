"""
Climb Generator API - Main Application (MVP)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import walls, generate
from app.database import init_db

app = FastAPI(
    title="Climb Generator API",
    description="API for browsing climbing walls and DDPM-based climb generation",
    version="0.3.0-mvp",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers — only read-only walls + generate for MVP
app.include_router(walls.router, prefix="/api/v1/walls", tags=["walls"])
app.include_router(generate.router, prefix="/api/v1/walls/{wall_id}/generate", tags=["generate"])


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    init_db()


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
