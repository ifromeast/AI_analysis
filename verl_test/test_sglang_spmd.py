import sys
sys.path.append("/data2/zzd/rl_llm/verl")

import os

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.verl_engine import VerlEngine
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.torch_functional import pad_sequence_to_length
from utils import are_lists_similar


def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    

    # NOTE MODIFIED should provide backend=None to have nccl+gloo
    # torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    dist.init_process_group(timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def test_sglang_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()
    # fill rollout config
    max_prompt_length = 16
    max_response_length = 16

    # Initialize model and token
    local_cache_path = "/data2/zzd/.cache/verl"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")

    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "痛饮狂歌空度日",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    actor_model.to(torch.bfloat16)

    sampling_params = dict(
        n=1,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_new_tokens=max_response_length,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        ignore_eos=False,
    )

    tensor_parallel_size = 4
    device_mesh_kwargs = dict(mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=["dp", "tp", "pp"])
    inference_device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    print("building sglang rollout engine")
    llm = VerlEngine(
        model_path=local_model_path,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        device_mesh_cpu=inference_device_mesh_cpu["tp"],
        base_gpu_id=0,
        gpu_id_step=1,
    )

    llm.release_memory_occupation()
    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    batch_size = input_ids.size(0)

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_response_length,
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

    hf_response_tokens = tokenizer.batch_decode(response)
    print(f"hf response: {hf_response_tokens}")
    print(f"{sampling_params=}")
    idx_list = []
    batch_size = input_ids.shape[0]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))

    outputs = llm.generate(input_ids=idx_list, sampling_params=sampling_params)
    sglang_response_tokens = []

    for output in outputs:
        print(f"{output=}")
        generated_text = output["text"]
        sglang_response_tokens.append(generated_text)

    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens), "Strings differ more than 10%:\n"
    print("✅ Check Passed!")

    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor):
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

if __name__ == "__main__":
    test_sglang_spmd()