import torch
from vllm import LLM, SamplingParams

# Create prompts, the same across all ranks
prompts = [
    "奇变偶不变",
    "The president of the United States is",
    "大鹏一日同风起，",
    "The future of AI is",
]

# Create sampling parameters, the same across all ranks
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Use `distributed_executor_backend="external_launcher"` so that
# this llm engine/instance only creates one worker.

model_name = "/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,
    distributed_executor_backend="external_launcher",
    dtype="bfloat16",
    seed=1,
)

outputs = llm.generate(prompts, sampling_params)

# Use torch profiler to profile the model
if True:
    torch.backends.cudnn.benchmark = True
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=5,),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiles/vllm_spmd_profile"),
    )
    profiler.start()

    for _ in range(20):
        outputs = llm.generate(prompts, sampling_params)
        profiler.step()
    profiler.stop()


# all ranks will have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")