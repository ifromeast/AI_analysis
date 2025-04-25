
import sys
sys.path.append("/data/zzd/verl")
import os
os.environ["NCCL_DEBUG"] = "WARN"

import copy
import numpy as np
import torch
import torch.distributed as dist

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.distributed import initialize_global_process_group


def test_all_gather_data_proto():
    device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=[2, 2], mesh_dim_names=["dp", "tp"])

    global_rank = dist.get_rank()

    obs = torch.tensor([[1 * global_rank, 2 * global_rank + 1], [3 * global_rank, 4 * global_rank + 1]])
    # rank 0: [[0,1],[0,1]]
    # rank 1: [[1,3],[3,5]]
    # rank 2: [[2,5],[6,9]]
    # rank 3: [[3,7],[9,13]]

    labels = ["a", "b"] if global_rank % 2 == 0 else ["c", "d"]
    labels = np.array(labels, dtype=object)
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    all_gather_data_proto(data=data, process_group=device_mesh.get_group("dp"))

    if global_rank == 0 or global_rank == 2:
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device="cuda")
        expected_labels = ["a", "b", "a", "b"]
    elif global_rank == 1 or global_rank == 3:
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device="cuda")
        expected_labels = ["c", "d", "c", "d"]

    torch.testing.assert_close(data.batch["obs"], expected_obs, atol=0, rtol=0)
    assert (data.non_tensor_batch["labels"] == expected_labels).all()
    assert data.meta_info == {"info": "test_info"}
    print("test_all_gather_data_proto 1 passed!")

    data = DataProto.from_dict(tensors={"obs": obs.clone()}, non_tensors={"labels": copy.deepcopy(labels)}, meta_info={"info": "test_info"})
    all_gather_data_proto(data=data, process_group=device_mesh.get_group("tp"))
    if global_rank == 0 or global_rank == 1:
        expected_obs = torch.tensor([[0, 1], [0, 1], [1, 3], [3, 5]], device="cuda")
        expected_labels = ["a", "b", "c", "d"]
    elif global_rank == 2 or global_rank == 3:
        expected_obs = torch.tensor([[2, 5], [6, 9], [3, 7], [9, 13]], device="cuda")
        expected_labels = ["a", "b", "c", "d"]
    torch.testing.assert_close(data.batch["obs"], expected_obs, atol=0, rtol=0)
    assert (data.non_tensor_batch["labels"] == expected_labels).all()
    assert data.meta_info == {"info": "test_info"}
    print("test_all_gather_data_proto 2 passed!")


def test_vocab_parallel_entropy(profile=True):
    from megatron.core import parallel_state as mpu

    from verl.utils.debug import log_gpu_memory_usage
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
    from verl.utils.torch_functional import entropy_from_logits

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=2, 
        pipeline_model_parallel_size=1, 
        virtual_pipeline_model_parallel_size=None
    )

    # 获取各种并行大小
    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()
    print(f"tp_size: {tp_size}, pp_size: {pp_size}, dp_size: {dp_size}")

    batch_size = 2
    seqlen = 128
    vocab_size = 155136

    logits = torch.randn(batch_size * seqlen, vocab_size, device="cuda", requires_grad=True)
    target = torch.randint(low=0, high=vocab_size, size=(batch_size * seqlen,), device="cuda", dtype=torch.int64)

    # broadcast across tp
    dist.broadcast(logits, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    dist.broadcast(target, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    tp_rank = mpu.get_tensor_model_parallel_rank()
    print(f"tp_rank: {tp_rank}")
    vocab_size_per_tp = vocab_size // mpu.get_tensor_model_parallel_world_size()

    # get the local logits of each tp
    vocab_parallel_logits = (
        logits.clone().detach()[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp].requires_grad_()
    )
    logits.grad = None
    vocab_parallel_logits.grad = None

    log_gpu_memory_usage("begin")
    output_entropy = vocab_parallel_entropy(vocab_parallel_logits)
    log_gpu_memory_usage("after forward")
    grad_output = torch.randn_like(output_entropy)
    output_entropy.backward(grad_output)
    log_gpu_memory_usage("after backward")

    target_entropy = entropy_from_logits(logits)
    torch.testing.assert_close(output_entropy, target_entropy)
    target_entropy.backward(grad_output)
    torch.testing.assert_close(
        logits.grad[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp], vocab_parallel_logits.grad
    )
    # make sure logits is not altered
    torch.testing.assert_close(
        logits[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp], vocab_parallel_logits
    )

    if mpu.get_tensor_model_parallel_rank() == 0:
        print("test_vocab_parallel_entropy passed!")

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5,),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiles/tp_dp_profile"),
        )
        profiler.start()

        for _ in range(20):
            output_entropy = vocab_parallel_entropy(vocab_parallel_logits)
            output_entropy.backward(grad_output)
            if profile:
                profiler.step()
        profiler.stop()

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    local_rank, rank, world_size = initialize_global_process_group()
    test_all_gather_data_proto()
    test_vocab_parallel_entropy(profile=False)
