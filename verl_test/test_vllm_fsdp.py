import sys
sys.path.append("/data/zzd/verl")

import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams, LLM

# from verl.third_party.vllm import LLM
from verl.utils.distributed import initialize_global_process_group


def main():
    assert torch.cuda.is_available(), "CUDA must be present to run FSDP vLLM example"
    local_rank, rank, world_size = initialize_global_process_group()

    local_cache_path = "/data/zzd/.cache/verl"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "/nfs/models/Qwen/Qwen2-7B-Instruct"

    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
        actor_model.to(torch.bfloat16)

    max_prompt_length = 16
    response_length = 32
    preencode_prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]
    from verl.utils.torch_functional import pad_sequence_to_length

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True).cuda()
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True).cuda()

    from transformers import GenerationConfig

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        # max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        # renormalize_logits=True,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=False,
    )  # may OOM when use_cache = True
    seq = output.sequences
    response = seq[:, max_prompt_length:]

    print(f"hf response: {tokenizer.batch_decode(response)}")

    tensor_model_parallel_size = 4
    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )

    FSDP.set_state_dict_type(
        fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig()
    )

    state_dict = fsdp_model.state_dict()

    sampling_params = SamplingParams(
        temperature=0, top_p=1, n=1, max_tokens=response_length, logprobs=1, ignore_eos=True, detokenize=False
    )

    print(actor_model_config)
    llm = LLM(
        model=hdfs_path,
        tokenizer=tokenizer,
        # model_hf_config=actor_model_config,
        tensor_parallel_size=tensor_model_parallel_size,
        enforce_eager=True,
        dtype="bfloat16",
        load_format="dummy_dtensor",
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    # Warmup iterations
    for _ in range(10):
        torch.cuda.synchronize()
        llm.sync_model_weights(actor_weights=state_dict, load_format="dtensor")
        torch.cuda.synchronize()
        dist.barrier()

    start_time = time.time()
    llm.sync_model_weights(actor_weights=state_dict, load_format="dtensor")
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    idx_list = []
    batch_size = input_ids.shape[0]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    from verl.workers.rollout.vllm_rollout.vllm_rollout import _pre_process_inputs

    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))
    print("start generation")
    outputs = llm.generate(prompt_token_ids=idx_list, sampling_params=sampling_params, use_tqdm=False)
    vllm_output = outputs[0].cuda()
    if torch.distributed.get_rank() == 0:
        print(f"hf response: {tokenizer.batch_decode(response)}")
        print(f"vllm response: {tokenizer.batch_decode(vllm_output)}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
