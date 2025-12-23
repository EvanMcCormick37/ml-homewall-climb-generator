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
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
import json
from fastapi.responses import FileResponse

from app.schemas import (
    HoldDetail,
    WallCreate,
    WallDetail,
    WallListResponse,
    WallCreateResponse,
)
from app.services import wall_service

router = APIRouter()


@router.get(
    "",
    response_model=WallListResponse,
    summary="List all walls",
    description="Returns metadata for all walls (without hold details).",
)
async def list_walls():
    """Get all walls with basic metadata."""
    walls = wall_service.get_all_walls()
    return WallListResponse(walls=walls, total=len(walls))


@router.post(
    "",
    response_model=WallCreateResponse,
    status_code=201,
    summary="Create a new wall",
    description="Create a new wall with holdset, metadata, and photo.",
)
async def create_wall(
    name: str = Form(..., min_length=1, max_length=100),
    holds: str = Form(..., description="JSON array of hold objects"),
    photo: UploadFile = File(..., description="Wall photo (JPEG or PNG)"),
    dimensions: str = Form(None, description="Comma-separated 'width,height' in cm"),
    angle: int = Form(None, description="Wall angle in degrees from vertical"),
):
    """Create a new wall from holdset, metadata, and photo."""
    
    # Validate photo type
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported.",
        )
    
    # Parse holds JSON
    try:
        holds_data = json.loads(holds)
        holds_list = [HoldDetail(**hold) for hold in holds_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid holds data: {str(e)}")
    
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
        holds=holds_list,
        dimensions=dims,
        angle=angle,
    )
    
    # Create wall with photo
    try:
        wall_id = await wall_service.create_wall(wall_data, photo)
        return WallCreateResponse(id=wall_id, name=wall_data.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create wall: {str(e)}")

@router.get(
    "/{wall_id}",
    response_model=WallDetail,
    summary="Get wall details",
    description="Returns full wall details including holds.",
)
async def get_wall(wall_id: str):
    """Get detailed wall info including holds."""
    wall = wall_service.get_wall(wall_id)
    if wall is None:
        raise HTTPException(status_code=404, detail="Wall not found")
    return wall


@router.delete(
    "/{wall_id}",
    status_code=204,
    summary="Delete a wall",
    description="Delete a wall and all associated climbs, models, and photos.",
)
async def delete_wall(wall_id: str):
    """Delete a wall and all associated data."""
    success = wall_service.delete_wall(wall_id)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return None


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
async def get_wall_photo(wall_id: str):
    """Get wall photo."""
    photo_path = wall_service.get_photo_path(wall_id)
    if photo_path is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    ext = photo_path.suffix
    media_type = "image/jpeg" if ext == ".jpg" else "image/png"
    return FileResponse(photo_path, media_type=media_type)


@router.put(
    "/{wall_id}/photo",
    status_code=200,
    summary="Upload wall photo",
    description="Upload or replace the wall photo.",
)
async def upload_wall_photo(
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

    success = wall_service.replace_photo(wall_id, photo)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return {"message": "Photo uploaded successfully"}
