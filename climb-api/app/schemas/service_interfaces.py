class ClimbService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_climbs(
        self, 
        wall_id: str,
        setter: str | None = None,
        name_includes: str | None = None,
        holds_include: list[int] | None = None,
        tags_include: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        sortBy: SortBy = SortBy.DATE,
    )->list[Climb]:
        """List climbs from the database matching the filter-fields and sorted by sortBy"""
    
    def delete_climb(
        self, 
        climb_id: str,
        setter_id: str,
        override: False
    ):
        """Delete a climb. Only works if climb_id matches setter_id or if override is true"""

    def set_climb(
        self, 
        wall_id: str,
        climb_id: str,
        setter_id: str,
        name: str,
        grade: str,
        sequence: list[tuple[int,int]]
        tags: list[str]
    ):
        """Set a climb in the database with the given climb params."""
    
class WallService:
    def __init__(self):
        """Initialize Wall JSON storage, photo Storage"""
    
    def save_wall(
        wall_id: str,
        metadata: JSON,
    ):
        """Save wall metadata and wall ID"""

    def get_wall(self, 
        wall_id: str
    ) -> JSON:
        """Return wall-data JSON with holds"""

    def save_wall_photo(self, 
        wall_id: str,
        photo: Image
    ):
        """Save/replace the image associated with a wall."""
    
    def get_wall_photo(self, 
        wall_id: str
    )-> Image:
        """Get the photo associated with the wall."""


class ModelService:
    def __init__(self):
        """Initialize model storage"""
    
    def create_model(
        self, 
        wall_id: str,
        model_type: str,
        augment_data: bool,
        train_epochs: int,
    ) -> str:
        """Creates a model and starts an asynchronous training job, returning jobId"""
    
    def delete_model(
        self, 
        model_id: str,
    ):
        """Deletes a model"""
    
    def generate_sequences(
        self, 
        model_id: str,
        start_holds: tuple[int,int]
        num_sequences: int = 1,
        temperature: float = .01,
        force_movement: bool = True,
    ) -> list[list[tuple[int,int]]]:
        """Generates num_sequences hold sequences from a given wall, and given pair of starting holds."""

class JobService:
    def __init__(self, db_manager: DatabaseManager):
        """Initialize jobs db"""
    
    def create_job(self, job_id: str) -> JobStatus:
        """Initialize a new job"""
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Return job status"""
    
    def complete_job(self, job_id: str):
        """Complete job"""
    
    def fail_job(self, job_id: str):
        """Fail job"""
    
    def delete_job(self, job_id: str):
        """Delete job"""