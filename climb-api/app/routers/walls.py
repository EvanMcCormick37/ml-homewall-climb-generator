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
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse

from app.schemas import (
    WallCreate,
    WallDetail,
    WallListResponse,
    WallCreateResponse,
)
from app.services.wall_service import WallService

router = APIRouter()
wall_service = WallService()


@router.get(
    "",
    response_model=WallListResponse,
    summary="List all walls",
    description="Returns metadata for all walls (without hold details).",
)
async def list_walls():
    """Get all walls with basic metadata."""
    # TODO: Implement
    # walls = wall_service.get_all_walls()
    # return WallListResponse(walls=walls, total=len(walls))
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post(
    "",
    response_model=WallCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new wall",
    description="Create a new wall with holdset and metadata.",
)
async def create_wall(wall_data: WallCreate):
    """Create a new wall from holdset and metadata."""
    # TODO: Implement
    # wall_id = wall_service.create_wall(wall_data)
    # return WallCreateResponse(id=wall_id, name=wall_data.name, num_holds=len(wall_data.holds))
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get(
    "/{wall_id}",
    response_model=WallDetail,
    summary="Get wall details",
    description="Returns full wall details including holds.",
)
async def get_wall(wall_id: str):
    """Get detailed wall info including holds."""
    # TODO: Implement
    # wall = wall_service.get_wall(wall_id)
    # if not wall:
    #     raise HTTPException(status_code=404, detail="Wall not found")
    # return wall
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete(
    "/{wall_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a wall",
    description="Delete a wall and all associated climbs, models, and photos.",
)
async def delete_wall(wall_id: str):
    """Delete a wall and all associated data."""
    # TODO: Implement
    # success = wall_service.delete_wall(wall_id)
    # if not success:
    #     raise HTTPException(status_code=404, detail="Wall not found")
    # return None
    raise HTTPException(status_code=501, detail="Not implemented")


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
    # TODO: Implement
    # photo_path = wall_service.get_photo_path(wall_id)
    # if not photo_path or not photo_path.exists():
    #     raise HTTPException(status_code=404, detail="Photo not found")
    # return FileResponse(photo_path, media_type="image/jpeg")
    raise HTTPException(status_code=501, detail="Not implemented")


@router.put(
    "/{wall_id}/photo",
    status_code=status.HTTP_200_OK,
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
    
    # TODO: Implement
    # success = wall_service.save_photo(wall_id, photo)
    # if not success:
    #     raise HTTPException(status_code=404, detail="Wall not found")
    # return {"message": "Photo uploaded successfully"}
    raise HTTPException(status_code=501, detail="Not implemented")
