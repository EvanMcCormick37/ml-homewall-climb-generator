"""
Pydantic schemas for climb-related requests and responses.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from app.schemas.base import Holdset


class ClimbSortBy(str, Enum):
    DATE = "date"
    NAME = "name"
    DIFFICULTY = "difficulty"
    ASCENTS = "ascents"


class Climb(BaseModel):
    id: str
    layout_id: str
    angle: int
    name: str
    holdset: Holdset
    difficulty: float | None = None
    quality: float | None = None
    ascents: int
    setter_name: str | None = None
    setter_id: str | None = None
    tags: list[str] | None = None
    created_at: datetime


class ClimbCreate(BaseModel):
    name: str
    holdset: Holdset
    angle: int | None
    difficulty: float | None = Field(None)
    quality: float | None = Field(2.5, ge=0, le=4)
    ascents: int | None = Field(0, ge=0)
    setter_name: str | None = None
    setter_id: str | None = None
    tags: list[str] | None = None


class ClimbBatchCreate(BaseModel):
    climbs: list[ClimbCreate]


class ClimbCreateResponse(BaseModel):
    id: str


class ClimbBatchCreateResult(BaseModel):
    index: int
    id: str | None = None
    status: str
    error: str | None = None


class ClimbBatchCreateResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: list[ClimbBatchCreateResult]


class ClimbDeleteResponse(BaseModel):
    id: str


class ClimbListResponse(BaseModel):
    climbs: list[Climb]
    total: int
    limit: int
    offset: int
