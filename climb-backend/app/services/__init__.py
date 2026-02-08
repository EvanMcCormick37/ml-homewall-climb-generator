"""
Service layer initialization with dependency injection.
"""
from app.services.container import ServiceContainer
from app.services import wall_service, climb_service, generation_service


# Build the container with all dependencies wired up
services = ServiceContainer(
    # Wall functions
    wall_exists=wall_service.wall_exists,
    get_num_holds=wall_service.get_num_holds,
    get_all_walls=wall_service.get_all_walls,
    get_wall=wall_service.get_wall,
    create_wall=wall_service.create_wall,
    delete_wall=wall_service.delete_wall,
    set_holds=wall_service.set_holds,
    get_photo_path=wall_service.get_photo_path,
    replace_photo=wall_service.replace_photo,
    
    # Climb functions
    get_climbs=climb_service.get_climbs,
    create_climb=climb_service.create_climb,
    create_climbs_batch=climb_service.create_climbs_batch,
    delete_climb=climb_service.delete_climb,
    get_climbs_for_training=climb_service.get_climbs_for_training,
    
    # Generation
    generate_climbs=generation_service.generate_climbs,
)

__all__ = ["services"]
