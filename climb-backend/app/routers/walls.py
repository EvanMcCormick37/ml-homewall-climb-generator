"""
Router for wall-related endpoints.

Endpoints:
- GET  /walls              - List all walls
- POST /walls              - Create a new wall
- GET  /walls/{wall_id}    - Get wall details
- DELETE /walls/{wall_id}  - Delete a wall
- GET  /walls/{wall_id}/photo    - Get wall photo
- PUT  /walls/{wall_id}/photo    - Upload/replace wall photo
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import json
from fastapi.responses import FileResponse

from app.schemas import (
    HoldDetail,
    WallCreate,
    WallDetail,
    WallListResponse,
    WallCreateResponse,
    SetHoldsResponse,
)
from app.services import services

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

@router.post(
    "",
    response_model=WallCreateResponse,
    status_code=201,
    summary="Create a new wall",
    description="Create a new wall with holdset, metadata, and photo.",
)
def create_wall(
    name: str = Form(..., min_length=1, max_length=100),
    photo: UploadFile = File(..., description="Wall photo (JPEG or PNG)"),
    dimensions: str = Form(None, description="Comma-separated 'width,height' in feet"),
    angle: int = Form(None, description="Wall angle in degrees from vertical"),
):
    """Create a new wall from holdset, metadata, and photo."""
    
    # Validate photo type
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported.",
        )
    
    # Parse dimensions
    dims = None
    if dimensions:
        try:
            parts = dimensions.split(",")
            dims = (int(parts[0].strip()), int(parts[1].strip()))
        except:
            raise HTTPException(status_code=400, detail="Invalid dimensions format. Use 'width,height'")
    
    # Create wall data object
    wall_data = WallCreate(
        name=name,
        dimensions=dims,
        angle=angle,
    )
    
    # Create wall with photo
    try:
        wall_id = services.create_wall(wall_data, photo)
        return WallCreateResponse(id=wall_id, name=wall_data.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create wall: {str(e)}")

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
        }
    )

@router.get(
    "/{wall_id}",
    response_model=WallDetail,
    summary="Get wall details",
    description="Returns full wall details including holds.",
)
def get_wall(wall_id: str):
    """Get detailed wall info including holds."""
    wall = services.get_wall(wall_id)
    if wall is None:
        raise HTTPException(status_code=404, detail="Wall not found")
    return wall

@router.delete(
    "/{wall_id}",
    status_code=204,
    summary="Delete a wall",
    description="Delete a wall and all associated climbs, models, and photos.",
)
def delete_wall(wall_id: str):
    """Delete a wall and all associated data."""
    success = services.delete_wall(wall_id)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return None

@router.put(
        "/{wall_id}/holds",
        status_code=201,
        summary="Add or replace a wall's holdset",
        description="Add or replace the holdset of a wall with a new list of holds. Hold data includes pixel_x, pixel_y, norm_x, norm_y, pull_x, pull_y, useability"
)
def set_holds(
    wall_id: str,
    holds: str = Form(..., description="JSON array of hold objects"),
    ) -> SetHoldsResponse :
    """Set or replace virtual holds on an existing wall."""
    # Parse holds JSON
    holds_data = json.loads(holds)
    holds_list = [HoldDetail(**hold) for hold in holds_data]
    try:
        success = services.set_holds(wall_id, holds_list)
        if not success:
            raise HTTPException(status_code=404, detail="Wall not found")
        return SetHoldsResponse(id=wall_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Attempting to set holds on {wall_id} resulted in Exception: {str(e)}")

@router.put(
    "/{wall_id}/photo",
    status_code=200,
    summary="Upload wall photo",
    description="Upload or replace the wall photo.",
)
def upload_wall_photo(
    wall_id: str,
    photo: UploadFile = File(..., description="Wall photo (JPEG or PNG)"),
):
    """Upload or replace wall photo."""
    # Validate file type
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported.",
        )

    success = services.replace_photo(wall_id, photo)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return {"message": "Photo uploaded successfully"}
