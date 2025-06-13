
"""
torchrun --standalone --nnodes=1 --nproc_per_node=2 test_sglang_async_rollout_without_tools.py
"""

import numpy as np
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

from verl import DataProto
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager


def test_async_sglang_rollout_without_tool():
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 32
    max_response_length = 16
    dtype = "bfloat16"
    tensor_parallel_size = 1
    local_model_path = "/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = [
        [{"role": "user", "content": prompt, "tool_calls": None}]
        for prompt in [
            "Who won the Champions League in 2019?",
            "The founder of Apple is",
            "What's the best way to learn python?",
        ]
    ]
    prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, "./sandbox_fusion_tool_config")
    rollout = SGLangRollout(actor_module=local_model_path, config=rollout_config, tokenizer=tokenizer, model_hf_config=actor_model.config)

    prompt_dict = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )
    print(f"preprocessed {input_ids.shape=}")

    messages = np.asarray(preencode_prompts)
    prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": np.array([{}] * input_ids.shape[0], dtype=object)})

    prompts.meta_info.update(
        {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    output = rollout.generate_sequences(prompts=prompts)
    print(f"generated {output.batch['responses'].shape}")
    sglang_output = output.to("cpu")

    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])

    print(f"sglang response: {sglang_response_tokens}")
    print("✅ SGLang w/o tool Test Passed!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_async_sglang_rollout_without_tool()
