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


@router.post(
    "",
    response_model=SizeCreateResponse,
    status_code=201,
    summary="Add a size to a layout (owner only)",
)
def create_size(
    layout_id: str,
    name: str = Form(..., min_length=1, max_length=100),
    edges: list[float] = Form(...),
    kickboard: bool = Form(...),
    user: dict = Depends(require_auth),
):
    """Create a new size for a layout, optionally uploading a photo."""
    if not services.layout_exists(layout_id):
        raise HTTPException(status_code=404, detail="Layout not found")
    
    size_data = SizeCreate(
        name=name,
        edges=edges,
        kickboard=kickboard,
    )

    try:
        size_id = services.create_size(layout_id, size_data)
        return SizeCreateResponse(id=size_id, layout_id=layout_id, name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create size: {e}")





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
