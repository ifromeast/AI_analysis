
import sys
sys.path.append("/data2/zzd/rl_llm/verl")
import pdb

import warnings
warnings.filterwarnings("ignore")
import ray
import torch
ray.init(
        runtime_env={
        "working_dir": "/data2/zzd/rl_llm/verl",  # 工作目录（会上传到集群）
    }
)

def get_config():
    from omegaconf import OmegaConf
    config = OmegaConf.load("/data2/zzd/rl_llm/verl/verl/trainer/config/ppo_trainer.yaml")
    config.data.train_files="/data2/zzd/data/full_hh_rlhf/rl/train.parquet"
    config.data.max_prompt_length=128
    config.data.filter_overlong_prompts=True
    config.actor_rollout_ref.model.path="/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"
    return config

config = get_config()

def get_test_data():
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn 
    from verl import DataProto
    from torchdata.stateful_dataloader import StatefulDataLoader

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    train_dataloader = StatefulDataLoader(
                dataset=train_dataset,
                batch_size=config.data.get("gen_batch_size", 8),
                num_workers=config.data.get("dataloader_num_workers", 1),
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_sampler,
            )
    for batch_dict in train_dataloader:
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        break
    return batch, gen_batch


from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.single_controller.ray import RayWorkerGroup

actor_rollout_mode = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
actor_worker = ray.remote(actor_rollout_mode)

resource_pool = RayResourcePool([8], use_gpu=True, max_colocate_count=1)
actor_cls = RayClassWithInitArgs(actor_worker, config=config.actor_rollout_ref, role="actor_rollout",)
actor_worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=actor_cls)

print("world size:", actor_worker_group.world_size)
print("worker_names:", actor_worker_group.worker_names)
actor_worker_group.init_model()

batch, gen_batch = get_test_data()
print(gen_batch)
gen_batch_output = actor_worker_group.generate_sequences(gen_batch)
print("gen_batch_output:", gen_batch_output)

# batch = batch.union(gen_batch_output)
# print("batch:", batch)

ray.shutdown()