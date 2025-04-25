import sys
sys.path.append("/data/zzd/verl")

import torch
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
from torch import nn
import torch.nn.functional as F

from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import logprobs_from_logits_naive


def test_flash_attn_cross_entropy():
    log_gpu_memory_usage("At start")

    hidden_states = torch.randn(size=(2048, 5120), device="cuda", requires_grad=True, dtype=torch.float32)
    linear = nn.Linear(in_features=5120, out_features=155136, bias=False, device="cuda", dtype=torch.float32)
    logits = linear(hidden_states)  # (2048, 155136)

    labels = torch.randint(low=0, high=155136, size=(2048,), device="cuda")

    log_gpu_memory_usage("before computation")
    output = cross_entropy_loss(logits, labels)[0]
    log_gpu_memory_usage("After forward")

    output.sum().backward()
    log_gpu_memory_usage("After backward")

    groundtruth = -logprobs_from_logits_naive(logits.float(), labels)
    torch.testing.assert_close(output, groundtruth)

    loss = F.cross_entropy(logits, labels)
    torch.testing.assert_close(loss, output.mean())
    log_gpu_memory_usage("After loss")


if __name__ == "__main__":
    test_flash_attn_cross_entropy()