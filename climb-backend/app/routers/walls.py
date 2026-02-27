"""
Router for wall-related endpoints (MVP — read-only).

Endpoints:
- GET  /walls              - List all walls
- GET  /walls/{wall_id}    - Get wall details
- GET  /walls/{wall_id}/photo    - Get wall photo
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from app.schemas import (
    WallDetail,
    WallListResponse,
)
from app.services import services
from app.auth import sync_auth

router = APIRouter()


@router.get(
    "",
    response_model=WallListResponse,
    summary="List all walls",
    description="Returns metadata for all walls (without hold details).",
)
def list_walls():
    """Get all walls with basic metadata."""
    walls = services.get_all_walls()
    return WallListResponse(walls=walls, total=len(walls))


@router.get(
    "/{wall_id}/photo",
    response_class=FileResponse,
    summary="Get wall photo",
    description="Returns the wall photo as a binary image.",
    responses={
        200: {"content": {"image/jpeg": {}, "image/png": {}}},
        404: {"description": "Wall or photo not found"},
    },
)
def get_wall_photo(wall_id: str):
    """Get wall photo."""
    photo_path = services.get_photo_path(wall_id)
    if photo_path is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    ext = photo_path.suffix
    media_type = "image/jpeg" if ext == ".jpg" else "image/png"
    return FileResponse(
        photo_path,
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get(
    "/{wall_id}",
    response_model=WallDetail,
    summary="Get wall details",
    description="Returns full wall details including holds.",
)
def get_wall(wall_id: str, _=Depends(sync_auth)):
    """Get detailed wall info including holds."""
    wall = services.get_wall(wall_id)
    if wall is None:
        raise HTTPException(status_code=404, detail="Wall not found")
    return wall
