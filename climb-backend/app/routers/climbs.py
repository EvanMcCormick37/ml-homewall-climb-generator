"""
Router for climb-related endpoints.
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends

from app.schemas import (
    ClimbCreate, ClimbSortBy, ClimbListResponse,
    ClimbCreateResponse, ClimbBatchCreate,
    ClimbBatchCreateResult, ClimbBatchCreateResponse, ClimbDeleteResponse,
)
from app.services import services
from app.auth import sync_auth, require_auth, get_accessible_wall, require_wall_owner

router = APIRouter()


@router.get("", response_model=ClimbListResponse, summary="List climbs")
def list_climbs(
    wall_id: str,
    angle: int | None = Query(None),
    grade_scale: str = Query("v_grade"),
    min_grade: str = Query("V0-"),
    max_grade: str = Query("V15"),
    include_projects: bool = Query(True),
    setter_name: str | None = Query(None),
    name_includes: str | None = Query(None),
    holds_include: list[int] | None = Query(None),
    tags_include: list[str] | None = Query(None),
    after: datetime | None = Query(None),
    sort_by: ClimbSortBy = Query(ClimbSortBy.DATE),
    descending: bool = Query(True),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    climbs, total, limit, offset = services.get_climbs(
        wall_id=wall_id,
        angle=angle,
        grade_scale=grade_scale,
        min_grade=min_grade,
        max_grade=max_grade,
        include_projects=include_projects,
        setter_name=setter_name,
        name_includes=name_includes,
        holds_include=holds_include,
        tags_include=tags_include,
        after=after,
        sort_by=sort_by,
        descending=descending,
        limit=limit,
        offset=offset,
    )
    return ClimbListResponse(climbs=climbs, total=total, limit=limit, offset=offset)


@router.post("", response_model=ClimbCreateResponse, status_code=201, summary="Create a climb")
def create_climb(
    wall_id: str,
    climb_create: ClimbCreate,
    _= Depends(require_auth),
):
    """Save a climb to a wall. Requires authentication + wall access."""
    if not services.wall_exists(wall_id):
        raise HTTPException(status_code=404, detail="Wall not found")
    try:
        climb_id = services.create_climb(wall_id, climb_create)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return ClimbCreateResponse(id=climb_id)


@router.post("/batch", response_model=ClimbBatchCreateResponse, status_code=201)
def create_climbs_batch(
    wall_id: str,
    batch_data: ClimbBatchCreate,
    _= Depends(require_auth),
):
    if not services.wall_exists(wall_id):
        raise HTTPException(status_code=404, detail="Wall not found")
    if not batch_data.climbs:
        raise HTTPException(status_code=400, detail="No climbs provided")

    results = services.create_climbs_batch(wall_id, batch_data.climbs)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    return ClimbBatchCreateResponse(
        total=len(results),
        successful=successful,
        failed=failed,
        results=[ClimbBatchCreateResult(**r) for r in results],
    )


@router.delete("/{climb_id}", response_model=ClimbDeleteResponse, summary="Delete a climb")
def delete_climb(
    wall_id: str,
    climb_id: str,
    _=Depends(require_wall_owner),
):
    """Delete a climb. Wall owner only."""
    success = services.delete_climb(wall_id, climb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Climb not found")
    return ClimbDeleteResponse(id=climb_id)