import sys
sys.path.append("/data2/zzd/rl_llm/verl")

import json
from typing import Any, Tuple

import numpy as np
import pytest
import ray
from omegaconf import DictConfig, OmegaConf
from transformers.utils import get_json_schema

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.utils import hf_tokenizer


def init_config() -> DictConfig:
    config = OmegaConf.load("/data2/zzd/rl_llm/verl/verl/trainer/config/ppo_trainer.yaml")
    model_path = "/data3/ckpt/Qwen/Qwen2.5-7B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_vllm_async_rollout_without_tool_calls(init_config):
    ray.shutdown()  # Ensure a clean start
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            },
            "working_dir": "/data2/zzd/rl_llm/verl",
        }
    )

    # =========================== 1. Init rollout manager ===========================
    async_rollout_manager = init_async_rollout_manager(init_config)

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [{"role": "user", "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",}],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(non_tensor_batch={"raw_prompt": np.array(raw_prompts),},)
    result = async_rollout_manager.generate_sequences(prompts=batch)
    print(result.batch["responses"])

    # decode responses
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    for i in range(len(responses)):
        valid_tokens = responses[i][response_mask[i].bool()]
        response_str = tokenizer.decode(valid_tokens)
        print(f"response {i}: {response_str}")

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert len(result) == 2
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()


class WeatherTool(BaseTool):
    def get_current_temperature(self, location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return json.dumps(result), 0, {}
        except Exception as e:
            return str(e), 0, {}


class WeatherToolWithDate(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_temperature_date)
        return OpenAIFunctionToolSchema(**schema)

    def get_temperature_date(self, location: str, date: str, unit: str = "celsius"):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        return {
            "temperature": 25.9,
            "location": location,
            "date": date,
            "unit": unit,
        }

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            result = self.get_temperature_date(**parameters)
            return json.dumps(result), 0, {}
        except Exception as e:
            return str(e), 0, {}


def test_vllm_async_rollout_with_tool_calls(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            },
            "working_dir": "/data2/zzd/rl_llm/verl",
        }
    )

    # =========================== 1. Init rollout manager ===========================
    tool_config = {
        "tools": [{"class_name": "test_vllm_async.WeatherTool", "config": {},},
                  {"class_name": "test_vllm_async.WeatherToolWithDate", "config": {},},]
    }
    tool_config_path = "./tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    async_rollout_manager = init_async_rollout_manager(init_config)

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [{"role": "user", "content": "How are you?"},],
        [{"role": "user", "content": "What's the temperature in Los Angeles now?"},],
        [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-21"},
         {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},],
    ]
    batch = DataProto(
        non_tensor_batch={"raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),},
    )
    result = async_rollout_manager.generate_sequences(prompts=batch)

    # Check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    # [user, assistant]
    assert num_turns[0] == 2
    # [user, assistant, tool, assistant]
    assert num_turns[1] == 4
    # [system, user, assistant, tool, tool, assistant]
    assert num_turns[2] == 6

    # Check response_mask
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"

    # Decode responses with response_mask
    for i in range(len(responses)):
        valid_tokens = responses[i][response_mask[i].bool()]
        response_str = tokenizer.decode(valid_tokens)
        assert "<tool_response>" not in response_str, f"found <tool_response> in response: {response_str}"
        assert "</tool_response>" not in response_str, f"found </tool_response> in response: {response_str}"
        print(f"response: {response_str}")

    print("✅ Test passed!")
    ray.shutdown()

if __name__ == "__main__":
    init_config = init_config()
    # test_vllm_async_rollout_without_tool_calls(init_config)
    test_vllm_async_rollout_with_tool_calls(init_config)
