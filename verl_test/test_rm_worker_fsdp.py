
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
    config.reward_model.model.input_tokenizer=None
    config.reward_model.model.path="/data3/ckpt/sfairXC/FsfairX-LLaMA3-RM-v0.1"
    config.reward_model.micro_batch_size_per_gpu=4
    return config

config = get_config()


def gen_test_batch_data():
    from verl import DataProto
    batch_dict = {
        "input_ids": torch.randint(0, 100, (16, 128)),
        "attention_mask": torch.ones((16, 128)),
        "position_ids": torch.arange(128).expand(16, -1),
        "responses": torch.randint(0, 100, (16, 32)),
    }
    batch = DataProto.from_single_dict(batch_dict)
    return batch


from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool
from verl.workers.fsdp_workers import RewardModelWorker
from verl.single_controller.ray import RayWorkerGroup

rm_worker = ray.remote(RewardModelWorker)

resource_pool = RayResourcePool([8], use_gpu=True, max_colocate_count=1)
rm_cls = RayClassWithInitArgs(rm_worker, config=config.reward_model)
rm_worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=rm_cls, name_prefix='rm')

print("world size:", rm_worker_group.world_size)
print("worker_names:", rm_worker_group.worker_names)
rm_worker_group.init_model()

batch = gen_test_batch_data()
reward_tensor = rm_worker_group.compute_rm_score(batch)
reward_tensor.to("cpu")
print(reward_tensor.batch['rm_scores'])

# Use torch profiler to profile the model
if False:
    torch.backends.cudnn.benchmark = True
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=5,),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiles/rm_worker_profile"),
    )
    profiler.start()

    for _ in range(20):
        reward_tensor = rm_worker_group.compute_rm_score(batch)
        profiler.step()
    profiler.stop()

ray.shutdown()