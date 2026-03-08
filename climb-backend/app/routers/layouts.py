"""
Router for layout-related endpoints.

A layout is a unique hold arrangement (replacing the old /walls concept).
"""
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from app.config import settings
import json

from app.schemas import (
    HoldDetail,
    LayoutDetail,
    LayoutListResponse,
    LayoutCreateResponse,
    LayoutCreate,
    LayoutEdit,
    SetHoldsResponse,
)
from app.services import services
from app.auth import require_auth, sync_auth, get_accessible_layout, require_layout_owner

router = APIRouter()


@router.get(
    "",
    response_model=LayoutListResponse,
    summary="List all layouts",
)
def list_layouts(user: dict | None = Depends(sync_auth)):
    """Returns public layouts + the authenticated user's own layouts."""
    user_id = user["user_id"] if user else None
    layouts = services.get_all_layouts(owner_id=user_id)
    return LayoutListResponse(layouts=layouts, total=len(layouts))


@router.get(
    "/{layout_id}",
    response_model=LayoutDetail,
    summary="Get layout details",
)
def get_layout(
    layout_id: str,
    _=Depends(get_accessible_layout),
):
    """Get detailed layout info including holds and sizes."""
    detail = services.get_layout(layout_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Layout not found")
    return detail


@router.post(
    "",
    response_model=LayoutCreateResponse,
    status_code=201,
    summary="Create a new layout",
)
def create_layout(
    name: str = Form(..., min_length=1, max_length=100),
    dimensions: str = Form(...),
    default_angle: int | None = Form(None),
    description: str | None = Form(None),
    visibility: str = Form("public"),
    image_edges: str = Form(...),
    user: dict = Depends(require_auth),
):
    """Create a new layout."""
    if visibility not in ("public", "private", "unlisted"):
        raise HTTPException(status_code=400, detail="Invalid visibility value.")

    layout_data = LayoutCreate(
        name=name,
        description=description,
        dimensions=json.loads(dimensions),
        default_angle=default_angle,
        image_edges=json.loads(image_edges),
        visibility=visibility
        )

    try:
        layout_id = services.create_layout(layout_data, owner_id=user["user_id"])
        return LayoutCreateResponse(id=layout_id, name=layout_data.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create layout: {e}")


@router.put(
    "/{layout_id}/edit",
    response_model=LayoutDetail,
    status_code=201,
    summary="Create a new layout",
)
def edit_layout(
    layout_id: str,
    name: str | None = Form(None, min_length=1, max_length=100),
    dimensions: str | None = Form(None),
    default_angle: int | None = Form(None),
    description: str | None = Form(None),
    visibility: str | None = Form(None),
    image_edges: str  | None = Form(None),
    # _=Depends(require_layout_owner),

):
    """Create a new layout."""
    if visibility is not None and visibility not in ("public", "private", "unlisted"):
        raise HTTPException(status_code=400, detail="Invalid visibility value.")

    layout_data = LayoutEdit(
        name=name,
        description=description,
        dimensions=json.loads(dimensions) if dimensions else None,
        default_angle=default_angle,
        image_edges=json.loads(image_edges) if image_edges else None,
        visibility=visibility
    )

    layout_id = services.put_layout(layout_id, layout_data)
    detail = services.get_layout(layout_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Layout not found")
    return detail


@router.put(
    "/{layout_id}/holds",
    response_model=SetHoldsResponse,
    status_code=201,
    summary="Set holds on a layout (owner only)",
)
def set_holds(
    layout_id: str,
    holds: str = Form(...),
    # _layout=Depends(require_layout_owner),
):
    """Set or replace the full hold set for a layout."""
    try:
        holds_data = json.loads(holds)
        holds_list = [HoldDetail(**hold) for hold in holds_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid holds JSON: {e}")

    success = services.set_holds(layout_id, holds_list)
    if not success:
        raise HTTPException(status_code=404, detail="Layout not found")
    return SetHoldsResponse(id=layout_id)


@router.delete(
    "/{layout_id}",
    status_code=200,
    summary="Delete a layout (owner only)",
)
def delete_layout(
    layout_id: str,
    # _layout=Depends(require_layout_owner),
):
    """Delete a layout and all its sizes, holds, and climbs."""
    success = services.delete_layout(layout_id)
    if not success:
        raise HTTPException(status_code=404, detail="Layout not found")
    return {"id": layout_id}


@router.put(
    "/{layout_id}/photo",
    status_code=200,
    summary="Upload or replace a size photo (owner only)",
)
def upload_layout_photo(
    layout_id: str,
    photo: UploadFile = File(...),
    # _: dict = Depends(require_auth),
):
    """Upload or replace the photo for a size."""
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")

    success = services.upload_layout_photo(layout_id, photo)
    if not success:
        raise HTTPException(status_code=404, detail="Size not found")
    return {"message": "Photo uploaded successfully"}


@router.get(
    "/{layout_id}/photo",
    response_class=FileResponse,
    summary="Get size photo",
)
def get_layout_photo(
    layout_id: str,
    _=Depends(get_accessible_layout),
):
    """Get the photo for a specific layout."""

    base_path = settings.LAYOUTS_DIR / layout_id / "photo"
    extensions = [".png", ".jpg", ".jpeg"]

    photo_path = None
    for ext in extensions:
        test_path = base_path.with_suffix(ext)
        if test_path.exists():
            photo_path = test_path
            break
    assert photo_path is not None

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
