"""
Router for wall-related endpoints.
"""
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
import json

from app.schemas import (
    HoldDetail,
    WallDetail,
    WallListResponse,
    WallCreateResponse,
    WallCreate,
    SetHoldsResponse,
)
from app.services import services
from app.auth import require_auth, sync_auth, get_accessible_wall, require_wall_owner

router = APIRouter()


@router.get(
    "",
    response_model=WallListResponse,
    summary="List all walls",
)
def list_walls(user: dict | None = Depends(sync_auth)):
    """
    Returns public walls + the authenticated user's own walls.
    Anonymous users see only public walls.
    """
    user_id = user["user_id"] if user else None
    walls = services.get_all_walls(owner_id=user_id)  # service filters accordingly
    return WallListResponse(walls=walls, total=len(walls))


@router.get(
    "/{wall_id}/photo",
    response_class=FileResponse,
    summary="Get wall photo",
)
def get_wall_photo(wall_id: str, wall=Depends(get_accessible_wall)):
    """Get wall photo — respects visibility."""
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
)
def get_wall(wall_id: str, _=Depends(get_accessible_wall)):
    """Get detailed wall info including holds — respects visibility."""
    full_wall = services.get_wall(wall_id)
    if full_wall is None:
        raise HTTPException(status_code=404, detail="Wall not found")
    return full_wall


@router.post(
    "",
    response_model=WallCreateResponse,
    status_code=201,
    summary="Create a new wall",
)
def create_wall(
    name: str = Form(..., min_length=1, max_length=100),
    photo: UploadFile = File(...),
    dimensions: str = Form(None),
    angle: int | None = Form(None),
    visibility: str = Form("public"),
    user: dict = Depends(require_auth),
):
    """Create a new wall. Requires authentication."""
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")

    if visibility not in ("public", "private", "unlisted"):
        raise HTTPException(status_code=400, detail="Invalid visibility value.")

    dims = None
    if dimensions:
        try:
            parts = dimensions.split(",")
            dims = (int(parts[0].strip()), int(parts[1].strip()))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid dimensions. Use 'width,height'.")

    wall_data = WallCreate(name=name, dimensions=dims, angle=angle, visibility=visibility)

    try:
        wall_id = services.create_wall(wall_data, photo, owner_id=user["user_id"])
        return WallCreateResponse(id=wall_id, name=wall_data.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create wall: {e}")


@router.put(
    "/{wall_id}/holds",
    status_code=201,
    summary="Set holds (owner only)",
)
def set_holds(
    wall_id: str,
    holds: str = Form(...),
    # _wall=Depends(require_wall_owner),
) -> SetHoldsResponse:
    """Set or replace holds. Owner only."""
    try:
        holds_data = json.loads(holds)
        holds_list = [HoldDetail(**hold) for hold in holds_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid holds JSON: {e}")

    success = services.set_holds(wall_id, holds_list)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return SetHoldsResponse(id=wall_id)


@router.put(
    "/{wall_id}/photo",
    status_code=200,
    summary="Upload wall photo (owner only)",
)
def upload_wall_photo(
    wall_id: str,
    photo: UploadFile = File(...),
    # _wall=Depends(require_wall_owner),
):
    """Upload or replace wall photo. Owner only."""
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")
    success = services.replace_photo(wall_id, photo)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return {"message": "Photo uploaded successfully"}


@router.delete(
    "/{wall_id}",
    status_code=200,
    summary="Delete a wall (owner only)",
)
def delete_wall(
    wall_id: str,
    # _wall=Depends(require_wall_owner),
):
    """Delete a wall and all its climbs. Owner only."""
    success = services.delete_wall(wall_id)
    if not success:
        raise HTTPException(status_code=404, detail="Wall not found")
    return {"id": wall_id}