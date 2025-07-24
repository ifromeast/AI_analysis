import sys

VERL_PATH = "/data2/zzd/rl_llm/verl"
sys.path.append(VERL_PATH)

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
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath(f"{VERL_PATH}/verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )
    model_path = "/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"
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

def test_tool_agent(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            },
            "working_dir": VERL_PATH,
        }
    )

    # =========================== 1. Init rollout manager ===========================
    tool_config = {
        "tools": [
            {
                "class_name": "rollout_test.test_vllm_async.WeatherTool",
                "config": {"type": "native"},
            },
            {
                "class_name": "rollout_test.test_vllm_async.WeatherToolWithDate",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/data2/zzd/tools/AI_analysis/verl_test/agent_test/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 2
    agent_loop_manager = init_agent_loop_manager(init_config)

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [{"role": "user", "content": "How are you?"},],
        [{"role": "user", "content": "What's the temperature in Los Angeles now?"},],
        [{"role": "user", "content": "What's the temperature in New York now?"},],
        [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30",},
         {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),
            "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # Check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    print(f"num_turns: {num_turns}")
    for i in range(len(num_turns)):
        if i // n == 0:
            # [user, assistant]
            assert num_turns[i] == 2
        else:
            # [user, assistant, tool, assistant]
            assert num_turns[i] == 4

    # Check response_mask
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"
    response_length = response_mask.size(1)

    # Decode responses with response_mask
    for i in range(len(responses)):
        valid_tokens = responses[i][response_mask[i][-response_length:].bool()]
        response_with_obs = tokenizer.decode(valid_tokens)

        # response without tool response
        valid_tokens = responses[i][response_mask[i].bool()]
        response_without_obs = tokenizer.decode(valid_tokens)

        assert "<tool_response>" not in response_without_obs, (f"found <tool_response> in response: {response_without_obs}")
        assert "</tool_response>" not in response_without_obs, (f"found </tool_response> in response: {response_without_obs}")
        print("============response_with_obs=============")
        print(response_with_obs)
        print("------------response_without_obs-------------")
        print(response_without_obs)

    print("Test passed!")
    ray.shutdown()


if __name__ == "__main__":
    init_config = init_config()
    test_tool_agent(init_config)
    print("✅ All tests passed!")
