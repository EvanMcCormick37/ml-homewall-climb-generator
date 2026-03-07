"""
Service layer initialization with dependency injection.
"""
from app.services.container import ServiceContainer
from app.services import (
    layout_service,
    size_service,
    climb_service,
    generation_service,
    user_service,
)


# Build the container with all dependencies wired up
services = ServiceContainer(
    # Layout functions (new)
    layout_exists=layout_service.layout_exists,
    get_all_layouts=layout_service.get_all_layouts,
    get_layout=layout_service.get_layout,
    get_layout_visibility=layout_service.get_layout_visibility,
    create_layout=layout_service.create_layout,
    delete_layout=layout_service.delete_layout,
    set_holds=layout_service.set_holds,
    upload_layout_photo=layout_service.upload_layout_photo,

    # Size functions (new)
    get_sizes=size_service.get_sizes,
    get_size=size_service.get_size,
    create_size=size_service.create_size,
    delete_size=size_service.delete_size,

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
