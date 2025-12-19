"""
Router for climb-related endpoints.

Endpoints:
- GET  /walls/{wall_id}/climbs             - List climbs with filters
- POST /walls/{wall_id}/climbs             - Create a new climb
- DELETE /walls/{wall_id}/climbs/{climb_id} - Delete a climb
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas import (
    Climb,
    ClimbCreate,
    ClimbSortBy,
    ClimbListResponse,
    ClimbCreateResponse,
    ClimbDeleteResponse,
)
from app.services.climb_service import ClimbService

router = APIRouter()
climb_service = ClimbService()


@router.get(
    "",
    response_model=ClimbListResponse,
    summary="List climbs",
    description="Get climbs for a wall with optional filtering and sorting.",
)
async def list_climbs(
    wall_id: str,
    name: str | None = Query(
        None, 
        description="Filter by name (partial match)"
    ),
    setter: str | None = Query(
        None, 
        description="Filter by setter ID"
    ),
    after: datetime | None = Query(
        None, 
        description="Filter climbs created after this date"
    ),
    sort_by: ClimbSortBy = Query(
        ClimbSortBy.DATE, 
        description="Sort order"
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
    includes_holds: str | None = Query(
        None,
        description="Comma-separated hold IDs that must be in the climb",
        example="1,5,12",
    ),
):
    """
    Get all climbs for a wall with filtering options.
    
    Filters:
    - name: Partial match on climb name
    - setter: Exact match on setter ID
    - after: Only climbs created after this datetime
    - includes_holds: Climbs must include ALL specified hold IDs
    """
    # Parse includes_holds from comma-separated string to list of ints
    hold_ids = None
    if includes_holds:
        try:
            hold_ids = [int(h.strip()) for h in includes_holds.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="includes_holds must be comma-separated integers",
            )
    
    # TODO: Implement
    # climbs, total = climb_service.get_climbs(
    #     wall_id=wall_id,
    #     name=name,
    #     setter=setter,
    #     after=after,
    #     sort_by=sort_by,
    #     limit=limit,
    #     offset=offset,
    #     includes_holds=hold_ids,
    # )
    # return ClimbListResponse(climbs=climbs, total=total, limit=limit, offset=offset)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post(
    "",
    response_model=ClimbCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a climb",
    description="Add a new climb to the wall.",
)
async def create_climb(wall_id: str, climb_data: ClimbCreate):
    """Create a new climb for a wall."""
    # TODO: Implement
    # Validate wall exists
    # if not wall_service.wall_exists(wall_id):
    #     raise HTTPException(status_code=404, detail="Wall not found")
    
    # Validate hold IDs exist in wall
    # ...
    
    # climb_id = climb_service.create_climb(wall_id, climb_data)
    # return ClimbCreateResponse(id=climb_id)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete(
    "/{climb_id}",
    response_model=ClimbDeleteResponse,
    summary="Delete a climb",
    description="Delete a climb by ID.",
)
async def delete_climb(wall_id: str, climb_id: str):
    """Delete a climb."""
    # TODO: Implement
    # success = climb_service.delete_climb(wall_id, climb_id)
    # if not success:
    #     raise HTTPException(status_code=404, detail="Climb not found")
    # return ClimbDeleteResponse(id=climb_id)
    raise HTTPException(status_code=501, detail="Not implemented")
