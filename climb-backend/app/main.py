"""
Climb Generator API - Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import walls, generate, climbs, layouts, sizes
from app.database import init_db

app = FastAPI(
    title="Climb Generator API",
    description="API for browsing climbing layouts and DDPM-based climb generation",
    version="0.4.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── New layout-based routes ───────────────────────────────────────────────────
app.include_router(layouts.router, prefix="/api/v1/layouts", tags=["layouts"])
app.include_router(sizes.router, prefix="/api/v1/layouts/{layout_id}/sizes", tags=["sizes"])

# Climbs and generate also work under /layouts/{layout_id}/...
app.include_router(
    generate.router,
    prefix="/api/v1/layouts/{layout_id}/generate",
    tags=["generate"],
)
app.include_router(
    climbs.router,
    prefix="/api/v1/layouts/{layout_id}/climbs",
    tags=["climbs"],
)

# ── Legacy wall-based routes (backward compat) ────────────────────────────────
app.include_router(walls.router, prefix="/api/v1/walls", tags=["walls"])
app.include_router(
    generate.router,
    prefix="/api/v1/walls/{layout_id}/generate",
    tags=["generate (legacy)"],
)
app.include_router(
    climbs.router,
    prefix="/api/v1/walls/{layout_id}/climbs",
    tags=["climbs (legacy)"],
)


@app.on_event("startup")
async def startup():
    """Initialize database and run migrations on startup."""
    init_db()


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy"}
