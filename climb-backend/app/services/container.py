"""
Dependency container for services.

Provides a central place to wire up service dependencies.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class ServiceContainer:
    """Container holding service functions with dependencies injected."""

    # Layout functions (new)
    layout_exists: Callable
    get_num_holds: Callable        # shared — works for both layout and wall IDs
    get_all_layouts: Callable
    get_layout: Callable
    get_layout_visibility: Callable
    create_layout: Callable
    delete_layout: Callable
    set_holds: Callable
    get_layout_photo_path: Callable

    # Size functions (new)
    get_sizes: Callable
    get_size: Callable
    create_size: Callable
    delete_size: Callable
    upload_size_photo: Callable
    get_size_photo_path: Callable

    # Wall functions (legacy — kept for /walls API backward compat)
    wall_exists: Callable
    get_all_walls: Callable
    get_wall: Callable
    get_wall_visibility: Callable
    create_wall: Callable
    delete_wall: Callable
    replace_photo: Callable

    # Climb functions
    get_climbs: Callable
    get_climb_setter_id: Callable
    create_climb: Callable
    create_climbs_batch: Callable
    delete_climb: Callable
    get_climbs_for_training: Callable

    # Generation
    generate_climbs: Callable

    # User Management
    ensure_user_exists: Callable
