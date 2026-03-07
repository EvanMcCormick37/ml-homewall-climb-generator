"""
Service layer initialization with dependency injection.
"""
from app.services.container import ServiceContainer
from app.services import (
    layout_service,
    size_service,
    wall_service,
    climb_service,
    generation_service,
    user_service,
)


# Build the container with all dependencies wired up
services = ServiceContainer(
    # Layout functions (new)
    layout_exists=layout_service.layout_exists,
    get_num_holds=layout_service.get_num_holds,
    get_all_layouts=layout_service.get_all_layouts,
    get_layout=layout_service.get_layout,
    get_layout_visibility=layout_service.get_layout_visibility,
    create_layout=layout_service.create_layout,
    delete_layout=layout_service.delete_layout,
    set_holds=layout_service.set_holds,
    get_layout_photo_path=layout_service.get_photo_path,

    # Size functions (new)
    get_sizes=size_service.get_sizes,
    get_size=size_service.get_size,
    create_size=size_service.create_size,
    delete_size=size_service.delete_size,
    upload_size_photo=size_service.upload_size_photo,
    get_size_photo_path=size_service.get_photo_path,

    # Wall functions (legacy — kept for /walls API backward compat)
    wall_exists=wall_service.wall_exists,
    get_all_walls=wall_service.get_all_walls,
    get_wall=wall_service.get_wall,
    get_wall_visibility=wall_service.get_wall_visibility,
    create_wall=wall_service.create_wall,
    delete_wall=wall_service.delete_wall,
    replace_photo=wall_service.replace_photo,

    # Climb functions
    get_climbs=climb_service.get_climbs,
    get_climb_setter_id=climb_service.get_climb_setter_id,
    create_climb=climb_service.create_climb,
    create_climbs_batch=climb_service.create_climbs_batch,
    delete_climb=climb_service.delete_climb,
    get_climbs_for_training=climb_service.get_climbs_for_training,

    # Generation
    generate_climbs=generation_service.generate_climbs,

    # User Management
    ensure_user_exists=user_service.ensure_user_exists,
)

__all__ = ["services"]
