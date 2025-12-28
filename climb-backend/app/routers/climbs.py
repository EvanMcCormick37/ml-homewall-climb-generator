"""
Router for climb-related endpoints.

Endpoints:
- GET  /walls/{wall_id}/climbs             - List climbs with filters
- POST /walls/{wall_id}/climbs             - Create a new climb
- DELETE /walls/{wall_id}/climbs/{climb_id} - Delete a climb
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    ClimbCreate,
    ClimbSortBy,
    ClimbListResponse,
    ClimbCreateResponse,
    ClimbDeleteResponse,
)
from app.services import services

router = APIRouter()


@router.get(
    "",
    response_model=ClimbListResponse,
    summary="List climbs",
    description="Get climbs for a wall with optional filtering and sorting.",
)
def list_climbs(
    wall_id: str,
    angle: int | None = Query(
        None,
        description="Filter by wall angle"
    ),
    grade_range: list[int] = Query(
        [0,180],
        min_length=2,
        max_length=2,
        description="min,max grade to filter climbs by"
    ),
    include_projects: bool = Query(
        True,
        description="Whether to include ungraded climbs."
    ),
    setter_name: str | None = Query(
        None, 
        description="Filter by setter name"
    ),
    name_includes: str | None = Query(
        None, 
        description="Filter by name (partial match)"
    ),
    holds_include: list[int] | None = Query(
        None,
        description="Comma-separated hold indices that must be in the climb",
        example="1,5,12",
    ),
    tags_include: list[str] | None = Query(
        None,
        description="Comma-separated tags that the climb must have",
        example="crimp,dynamic",
    ),
    after: datetime | None = Query(
        None, 
        description="Filter climbs created after this date"
    ),
    sort_by: ClimbSortBy = Query(
        ClimbSortBy.DATE, 
        description="Sort order"
    ),
    descending: bool = Query(
        True,
        description="Sort results descending?"
    ),
    limit: int = Query(
        50, 
        ge=1, 
        le=200, 
        description="Maximum number of results"
    ),
    offset: int = Query(
        0, 
        ge=0, 
        description="Offset for pagination"
    ),
):
    """
    Get all climbs for a wall with filtering options.
    
    Filters:
    - angle: Exact match on wall angle
    - name_includes: Partial match on climb name
    - setter_name: Exact match on setter name
    - after: Only climbs created after this datetime
    - holds_include: Climbs must include ALL specified hold indices
    - tags_include: Climbs must have ALL specified tags
    """
    climbs, total, limit, offset = services.get_climbs(
        wall_id=wall_id,
        angle=angle,
        grade_range=grade_range,
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

    return ClimbListResponse(
        climbs=climbs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=ClimbCreateResponse,
    status_code=201,
    summary="Create a climb",
    description="Add a new climb to the wall.",
)
def create_climb(wall_id: str, climb_data: ClimbCreate):
    """Create a new climb for a wall."""
    if not services.wall_exists(wall_id):
        raise HTTPException(status_code=404, detail="Wall not found")
    climb_id = services.create_climb(wall_id, climb_data)
    return ClimbCreateResponse(id=climb_id)


@router.delete(
    "/{climb_id}",
    response_model=ClimbDeleteResponse,
    summary="Delete a climb",
    description="Delete a climb by ID.",
)
def delete_climb(wall_id: str, climb_id: str):
    """Delete a climb."""
    success = services.delete_climb(wall_id, climb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Climb not found")
    return ClimbDeleteResponse(id=climb_id)