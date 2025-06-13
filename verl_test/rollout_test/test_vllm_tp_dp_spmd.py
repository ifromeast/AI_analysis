import sys
sys.path.append("/data2/zzd/rl_llm/verl")

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams

from verl.utils.distributed import initialize_global_process_group
from torch.distributed.device_mesh import init_device_mesh
from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout


def get_config():
    from omegaconf import OmegaConf
    config = OmegaConf.load("/data2/zzd/rl_llm/verl/verl/trainer/config/generation.yaml")
    config.data.path="/data2/zzd/data/full_hh_rlhf/rl/train.parquet"
    config.model.path="/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"
    config.rollout.tensor_model_parallel_size=4
    config.rollout.gpu_memory_utilization=0.8
    return config

config = get_config()

def get_test_data(tokenizer, rank):
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import pad_sequence_to_length

    max_prompt_length = 32
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        # "The founder of Apple is",
        # "痛饮狂歌空度日",
        # "13*24="
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]
    
    # position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).unsqueeze(0)

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)
    # position_ids = pad_sequence_to_length(position_ids, max_prompt_length, 0, left_pad=True)
    position_ids = compute_position_id_with_mask(attention_mask)

    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    position_ids = position_ids.cuda()

    data = DataProto.from_single_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })
    data.meta_info["eos_token_id"] = tokenizer.eos_token_id

    return data


def test_vllm_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and token
    local_cache_path = "/data2/zzd/.cache/verl"
    local_cache_path = os.path.expanduser(local_cache_path)
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=config.model.path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)

    infer_tp = config.rollout.tensor_model_parallel_size
    dp = world_size // infer_tp
    assert world_size % infer_tp == 0, f"rollout world_size: {world_size} is not divisible by infer_tp: {infer_tp}"
    rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])


    rollout = vLLMRollout(
        model_path=local_model_path,
        config=config.rollout,
        tokenizer=tokenizer,
        model_hf_config=hf_config,
        device_mesh=rollout_device_mesh,
        trust_remote_code=True,
    )

    prompts_data = get_test_data(tokenizer, rank)
    print("start generation")

    print(f"rank:{rank}, prompts_data: {prompts_data}")
    response = rollout.generate_sequences(prompts_data)
    print(f"rank:{rank}, response: {response}")


if __name__ == "__main__":
    test_vllm_spmd()
