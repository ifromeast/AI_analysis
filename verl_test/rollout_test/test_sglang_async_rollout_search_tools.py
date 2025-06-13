
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import torch
from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

from verl.protocol import DataProto
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
from verl.tools.search_tool import SearchTool
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + "How's the weather lately?"


def get_search_messages():
    user_prompt = {
        "role": "user",
        "content": user_content,
    }

    expect_turn_0_msg = {
        "role": "assistant",
        "content": "Let me search the web.",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "today's weather"}}}],
    }

    expect_turn_1_msg = {
        "role": "assistant",
        "content": "Let me search again.",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "tomorrow's weather"}}}],
    }

    expect_turn_2_msg = {
        "role": "assistant",
        "content": "<answer>Today is sunny and tomorrow will be cloudy in Beijing.</answer>",
    }

    # Mock search tool responses
    tool_return_0_msg = {"role": "tool", "content": "Today's weather in Beijing is sunny."}
    tool_return_1_msg = {"role": "tool", "content": "Tomorrow's weather in Beijing is cloudy."}

    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    return user_prompts, expect_turn_array, tool_return_array


def qwen_tokenizer_config(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(local_model_path)
    return tokenizer, config

def search_data(qwen_tokenizer):
    user_prompt, expect_turn_array, tool_return_array = get_search_messages()
    prompts = [[message] for message in user_prompt]
    preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
    preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
    return prompts, preencode_turn_array, preencode_tool_return_array


def search_rollout_config():
    max_prompt_length = 4096
    max_response_length = 3000
    dtype = "bfloat16"
    tensor_parallel_size = 1
    tool_path = "./search_tool_config"
    rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
    return rollout_config


def search_data_proto(search_data, qwen_tokenizer):
    preencode_prompts, _, _ = search_data
    prompts = [qwen_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
    input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
    prompt_dict = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )
    messages = np.asarray(preencode_prompts)

    tools_kwargs = np.array(
        [{"search": {"create_kwargs": {"ground_truth": "Today is sunny and tomorrow will be cloudy in Beijing.", "data_source": "searchR1_nq"},},}],
        dtype=object,
    )
    index = np.array([0], dtype=object)
    prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})
    return prompts


def test_tools(local_model_path, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, expect_turn_array, tool_return_array):
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    rollout = SGLangRollout(actor_module=local_model_path, config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
    assert len(rollout._tool_schemas) == 1
    assert "search" in rollout._tool_map.keys()
    assert isinstance(rollout._tool_map["search"], SearchTool)
    # depend on the tokenizer
    assert rollout._tool_call_parser_type == "qwen25"

    req_list = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
    assert len(req_list) == 1
    assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
    assert len(req_list[0].tool_schemas) == 1
    print(type(req_list[0].tool_schemas[0]))

    assert req_list[0].tool_schemas[0] == OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="search",
            description="Searches the web for relevant information based on the given query.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "query_list": OpenAIFunctionPropertySchema(
                        type="array",
                        description="A list of fully-formed semantic queries. The tool will return search results for each query.",
                        items={"type": "string"},
                    )
                },
                required=["query_list"],
            ),
            strict=False,
        ),
    )


    req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
    req_list = [req]

    rollout._handle_engine_call = MagicMock()
    future = asyncio.Future()
    future.set_result({"text": expect_turn_array[0], "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "length", "length": 3000}, "prompt_tokens": 132, "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
    rollout._handle_engine_call.return_value = future

    rollout._tp_rank = 0
    loop = asyncio.get_event_loop()
    output_req_list = loop.run_until_complete(
        asyncio.gather(
            *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
        )
    )
    assert len(output_req_list) == 1
    output_req = output_req_list[0]
    assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
    assert output_req.reward_scores == {"search": []}, f"output_req.reward_scores: {output_req.reward_scores}"
    # we should only have two message, one for prompt, second for response.
    assert len(output_req.messages) == 2
    # print(f"output_req.messages: {output_req.messages}")
    assert output_req.messages[1] == Message(
        role="assistant",
        content=expect_turn_array[0],
        tool_calls=None,
    )

    # print("req_list:  ", req_list)
    SearchTool.execute = AsyncMock()
    SearchTool.execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]
    rollout._handle_engine_call = MagicMock()
    futures = [asyncio.Future() for i in expect_turn_array]
    for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
        i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
        if idx < len(expect_turn_array) - 1:
            assert rollout._function_call_parser.has_tool_call(turn)
            assert rollout._function_call_parser.parse_non_stream(turn)

    rollout._handle_engine_call.side_effect = futures
    rollout._tp_rank = 0

    loop = asyncio.get_event_loop()
    output_req_list = loop.run_until_complete(asyncio.gather(*[rollout._async_rollout_a_request(req, True, False) for req in req_list]))

    # Verify conversation completed successfully with proper tool usage
    output_req = output_req_list[0]
    assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
    assert "search" in output_req.metrics
    assert output_req.metrics["search"][0]["status"] == "success"
    assert SearchTool.execute.await_count == 2
    assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
    # Verify tool response messages contain expected content
    # print(output_req.messages)
    search_counter = 0
    for msg in output_req.messages:
        if msg.role == "tool":
            assert msg.content == tool_return_array[search_counter]
            search_counter += 1
    assert search_counter == 2

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    local_model_path = "/data3/ckpt/Qwen/Qwen2.5-3B-Instruct"
    tokenizer, model_config = qwen_tokenizer_config(local_model_path)
    rollout_config = search_rollout_config()
    prompts, preencode_turn_array, preencode_tool_return_array = search_data(tokenizer)
    print(f"preencode_turn_array: {preencode_turn_array}")
    print(f"preencode_tool_return_array: {preencode_tool_return_array}")
    data_proto = search_data_proto((prompts, preencode_turn_array, preencode_tool_return_array), tokenizer)
    print(f"data_proto: {data_proto}")
    test_tools(local_model_path, rollout_config, tokenizer, model_config, data_proto, preencode_turn_array, preencode_tool_return_array)
