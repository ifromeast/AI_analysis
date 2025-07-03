
from typing import Union

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop import AgentLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


def init_agent_loop_manager(config: DictConfig) -> Union[AgentLoopManager, RayWorkerGroup]:
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    role_worker_mapping = {Role.ActorRollout: ray.remote(actor_rollout_cls),}

    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,}
    mapping = {Role.ActorRollout: global_pool_id,}
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout")
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    if config.actor_rollout_ref.rollout.mode == "sync":
        return actor_rollout_wg

    # =========================== 2. Create AgentLoopManager ===========================
    print("Creating AgentLoopManager...")
    agent_loop_manager = AgentLoopManager(
        config=config,
        worker_group=actor_rollout_wg,
    )

    return agent_loop_manager
