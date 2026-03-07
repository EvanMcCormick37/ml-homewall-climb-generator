"""
Router for size-related endpoints.

Sizes are physical variants of a layout (different dimensions / photo).
All size endpoints are nested under /layouts/{layout_id}/sizes.
"""
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse

from app.schemas.sizes import SizeMetadata, SizeCreate, SizeCreateResponse
from app.services import services
from app.auth import require_auth, get_accessible_layout

router = APIRouter()


@router.get(
    "",
    response_model=list[SizeMetadata],
    summary="List sizes for a layout",
)
def list_sizes(
    layout_id: str,
    _=Depends(get_accessible_layout),
):
    """List all sizes for a given layout."""
    return services.get_sizes(layout_id)


@router.get(
    "/{size_id}/photo",
    response_class=FileResponse,
    summary="Get size photo",
)
def get_size_photo(
    layout_id: str,
    size_id: str,
    _=Depends(get_accessible_layout),
):
    """Get the photo for a specific size."""
    photo_path = services.get_size_photo_path(layout_id, size_id)
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


@router.post(
    "",
    response_model=SizeCreateResponse,
    status_code=201,
    summary="Add a size to a layout (owner only)",
)
def create_size(
    layout_id: str,
    name: str = Form(..., min_length=1, max_length=100),
    width_ft: float | None = Form(None),
    height_ft: float | None = Form(None),
    edge_left: float = Form(0.0),
    edge_right: float | None = Form(None),
    edge_bottom: float = Form(0.0),
    edge_top: float | None = Form(None),
    photo: UploadFile | None = File(None),
    user: dict = Depends(require_auth),
):
    """Create a new size for a layout, optionally uploading a photo."""
    if not services.layout_exists(layout_id):
        raise HTTPException(status_code=404, detail="Layout not found")

    if photo and photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")

    size_data = SizeCreate(
        name=name,
        width_ft=width_ft,
        height_ft=height_ft,
        edge_left=edge_left,
        edge_right=edge_right,
        edge_bottom=edge_bottom,
        edge_top=edge_top,
    )

    try:
        size_id = services.create_size(layout_id, size_data, photo=photo)
        return SizeCreateResponse(id=size_id, layout_id=layout_id, name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create size: {e}")


@router.put(
    "/{size_id}/photo",
    status_code=200,
    summary="Upload or replace a size photo (owner only)",
)
def upload_size_photo(
    layout_id: str,
    size_id: str,
    photo: UploadFile = File(...),
    user: dict = Depends(require_auth),
):
    """Upload or replace the photo for a size."""
    if photo.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")

    success = services.upload_size_photo(layout_id, size_id, photo)
    if not success:
        raise HTTPException(status_code=404, detail="Size not found")
    return {"message": "Photo uploaded successfully"}


@router.delete(
    "/{size_id}",
    status_code=200,
    summary="Delete a size (owner only)",
)
def delete_size(
    layout_id: str,
    size_id: str,
    user: dict = Depends(require_auth),
):
    """Delete a size and its photo."""
    success = services.delete_size(size_id)
    if not success:
        raise HTTPException(status_code=404, detail="Size not found")
    return {"id": size_id}
