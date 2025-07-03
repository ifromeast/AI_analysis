import sys
sys.path.append("/data/zzd/verl")

import json
import os
from typing import Any, Tuple

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from transformers.utils import get_json_schema

from agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.utils import hf_tokenizer

def init_config() -> DictConfig:
    config = OmegaConf.load("/data/zzd/verl/verl/trainer/config/ppo_trainer.yaml")
    model_path = "/data/zzd/ckpt/Qwen/Qwen2.5-1.5B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.agent.num_workers = 2

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_single_turn(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            },
            "working_dir": "/data/zzd/verl",  # 工作目录（会上传到集群）
        }
    )

    print("Initializing agent loop manager...")
    agent_loop_manager = init_agent_loop_manager(init_config)

    raw_prompts = [
        [{"role": "user","content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",}],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
        },
    )

    print("Generating sequences by agent...")
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * init_config.actor_rollout_ref.rollout.n

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()


if __name__ == "__main__":
    init_config = init_config()
    test_single_turn(init_config)
    print("✅ All tests passed!")