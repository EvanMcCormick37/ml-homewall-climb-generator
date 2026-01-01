"""
Dependency container for services.

Provides a central place to wire up service dependencies.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class ServiceContainer:
    """Container holding service functions with dependencies injected."""
    
    # Job functions
    create_job: Callable
    get_job: Callable
    update_job_status: Callable
    complete_job: Callable
    fail_job: Callable
    
    # Wall functions
    wall_exists: Callable
    get_num_holds: Callable
    get_all_walls: Callable
    get_wall: Callable
    create_wall: Callable
    delete_wall: Callable
    set_holds: Callable
    get_photo_path: Callable
    replace_photo: Callable
    
    # Climb functions
    get_climbs: Callable
    create_climb: Callable
    create_climbs_batch: Callable
    delete_climb: Callable
    get_climbs_for_training: Callable
    
    # Model functions
    get_models_for_wall: Callable
    get_model: Callable
    create_model: Callable
    delete_model: Callable
    train_model_task: Callable
    generate_climbs: Callable