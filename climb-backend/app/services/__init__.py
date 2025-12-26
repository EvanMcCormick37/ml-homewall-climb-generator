"""
Service layer initialization with dependency injection.
"""
from app.services.container import ServiceContainer
from app.services import job_service, wall_service, climb_service, model_service


# Build the container with all dependencies wired up
services = ServiceContainer(
    # Job functions
    create_job=job_service.create_job,
    get_job=job_service.get_job,
    update_job_status=job_service.update_job_status,
    complete_job=job_service.complete_job,
    fail_job=job_service.fail_job,
    
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
    delete_climb=climb_service.delete_climb,
    get_climbs_for_training=climb_service.get_climbs_for_training,
    
    # Model functions
    get_models_for_wall=model_service.get_models_for_wall,
    get_model=model_service.get_model,
    create_model=model_service.create_model,
    delete_model=model_service.delete_model,
    train_model_task=model_service.make_train_model_task(
        on_update=job_service.update_job_status,
        on_complete=job_service.complete_job,
        on_fail=job_service.fail_job,
    ),
    generate_climbs=model_service.generate_climbs,
)


# For convenience, export individual functions if preferred
__all__ = ["services"]
