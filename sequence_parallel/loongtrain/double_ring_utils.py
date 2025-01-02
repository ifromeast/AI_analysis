#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context

# import inspect
# import random
# import socket
# import sys
# from importlib.machinery import SourceFileLoader
# from pathlib import Path
# from typing import Union

from typing import Dict, List, Union, Optional, Tuple
import math
import numpy as np
import torch
import torch.distributed as dist
import datetime
from functools import reduce


LLM_NCCL_TIMEOUT = datetime.timedelta(seconds=1800)

# class Singleton:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
#         return cls._instance


# class ProcessGroupSingleton(Singleton):
#     def __init__(self):
#         self.ULYSSES_PG = None
#         self.RING_PG = None


# PROCESS_GROUP = ProcessGroupSingleton()



class GroupConfig:
    """config for initialze a process group"""

    def __init__(
        self,
        mode,
        size: int,
        anonymous: bool = False,
        allow_partial_group: bool = False,
        subgroups: Optional[List["GroupConfig"]] = None,
    ) -> None:
        self.mode = mode
        self.size = size
        self.anonymous = anonymous
        self.allow_partial_group = allow_partial_group
        self.subgroups = subgroups if subgroups is not None else []

        self._early_subgroup_checking()

    def _early_subgroup_checking(self) -> None:
        if len(self.subgroups) == 0:
            return

        group_target_size = reduce(lambda x, y: x * y, [_g.size for _g in self.subgroups])
        assert group_target_size <= self.size, "subgroup size should less than father group"



def get_group_ranks(
    global_ranks_or_sizes: Union[int, List[int]],
    cur_group_size: int,
    pre_group_size: int,
    allow_partial_group: bool = False,
):
    group_ranks = []

    if isinstance(global_ranks_or_sizes, list):
        global_size = len(global_ranks_or_sizes)
        global_ranks = global_ranks_or_sizes
    else:
        global_size = global_ranks_or_sizes
        global_ranks = None

    real_global_size = global_size

    if allow_partial_group:
        global_size = math.ceil(global_size / cur_group_size) * cur_group_size

    assert global_size % cur_group_size == 0, "err1"

    def _get_local_starts():
        for i in range(0, global_size, cur_group_size * pre_group_size):
            for j in range(pre_group_size):
                yield 0 + i + j

    for start in _get_local_starts():
        ranks = [
            start + i * pre_group_size for i in range(cur_group_size) if start + i * pre_group_size < real_global_size
        ]
        if global_ranks is not None:
            ranks = [global_ranks[_idx] for _idx in ranks]

        group_ranks.append(ranks)

    assert len(group_ranks) == global_size // cur_group_size, f"{group_ranks}, {global_size}, {cur_group_size}"

    return group_ranks



def _create_parallel_process_groups(
    global_ranks_or_sizes: int,
    self_rank: int,
    pre_group_size: int,
    group_configs: List[GroupConfig],
    with_cpu_group: bool = False,
):
    group_results = []

    for group in group_configs:
        if group.anonymous is True:
            pre_group_size = pre_group_size * group.size
            continue

        group_ranks, accelerator_group = None, None
        all_group_ranks = get_group_ranks(global_ranks_or_sizes, group.size, pre_group_size, group.allow_partial_group)

        for idx, ranks in enumerate(all_group_ranks):
            _pg = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if self_rank in ranks:
                group_ranks, accelerator_group = all_group_ranks[idx], _pg
            else:
                dist.destroy_process_group(_pg)

        if group_ranks is None:
            pre_group_size = pre_group_size * group.size
            continue

        cpu_group = None
        group_results.append((group_ranks.index(self_rank), len(group_ranks), accelerator_group, cpu_group, group_ranks, group.mode))

        if len(group.subgroups) > 0:
            subgroup_results = _create_parallel_process_groups(global_ranks_or_sizes, self_rank, pre_group_size, group.subgroups, with_cpu_group=False)
            group_results.extend(subgroup_results)

        pre_group_size = pre_group_size * group.size

    return group_results





def generate_2d_attn_process_group(
    world_size: int,
    self_rank: int,
    head_size: int = 1,
    context_size: int = 1,
    window_size: int = 1,
    head_first: bool = True,
    interleaved: bool = False,
    sp_size: int = 1,
    with_cpu_group: bool = False,
):
    """
    head_size: 字段表示 head parallel size
    context_size: 字段表示 context parallel size
    window_size: 字段表示 Double-Ring Attention 中的 window_size
    head_first: 字段表示是否优先分配 head parallel 通信组，若为False，则为 context-first
    interleaved: 字段表示是否对 context parallel 的 GPU 重排
    """

    assert context_size * head_size == sp_size, "context_size * head_size should equal to sp_size"
    assert world_size % sp_size == 0, "world_size should be divisible by sp_size"

    if (window_size >= 8 or window_size == context_size) and interleaved:
        print("interleaved is forced False when window size > 8 or equals context size.")
        interleaved = False

    if head_first and head_size > 1 and interleaved:
        print("interleaved is forced False when head_first is True and head size > 1.")
        interleaved = False

    group_results = []
    sp_pre_group_size = 1

    # head and context process groups.
    if head_first:
        group_configs = [
            GroupConfig("head", head_size),
            GroupConfig("context", context_size),
        ]
        context_results_index = 1
    else:
        group_configs = [
            GroupConfig("context", context_size),
            GroupConfig("head", head_size),
        ]
        context_results_index = 0

    group_results.extend(_create_parallel_process_groups(world_size, self_rank, sp_pre_group_size, group_configs, with_cpu_group))

    # window process groups.
    window_num = context_size // window_size
    cp_pre_group_size = 1 if context_results_index == 0 else head_size
    every_context_ranks = get_group_ranks(world_size, context_size, cp_pre_group_size)

    def _gen_window_process_groups(context_ranks: List[int]):
        if not interleaved:
            window_ranks = context_ranks
        else:
            _indexes = [
                j * 2 + i * window_size if i % 2 == 0 else j * 2 + 1 + (i - 1) * window_size
                for i in range(window_num)
                for j in range(window_size)
            ]
            window_ranks = [context_ranks[_i] for _i in _indexes]

        group_results.extend(
            _create_parallel_process_groups(
                window_ranks,
                self_rank,
                1,
                [
                    GroupConfig("intra_window", window_size),
                    GroupConfig("inter_window", window_num),
                ],
                with_cpu_group,
            )
        )
        group_results.extend(
            _create_parallel_process_groups(
                window_ranks,
                self_rank,
                1,
                [
                    GroupConfig("dkv_intra_window", window_size),
                    GroupConfig("dkv_inter_window", window_num),
                ],
                with_cpu_group,
            )
        )

    for context_ranks in every_context_ranks:
        _gen_window_process_groups(context_ranks)

    # print(get_group_ranks(window_ranks, config.window_size, 1))
    # print(get_group_ranks(window_ranks, window_num, config.window_size))

    return group_results


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    res = generate_2d_attn_process_group(
                                    world_size,
                                    rank,
                                    head_size=2,
                                    context_size=4,
                                    window_size=4,
                                    head_first=True,
                                    interleaved=False,
                                    sp_size=8,
                                    with_cpu_group=False,
                                )
    # print(f"rank:{rank}, res:\n{res}\n\n")

    for item in res:
        if item[5] == "head":
            print(f"rank:{rank}, head group ranks: {item[4]}")
        elif item[5] == "context":
            print(f"rank:{rank}, context group ranks: {item[4]}")
        elif item[5] == "intra_window":
            print(f"rank:{rank}, intra_window group ranks: {item[4]}")
        elif item[5] == "inter_window":
            print(f"rank:{rank}, inter_window group ranks: {item[4]}")



    if dist.is_initialized():
        dist.destroy_process_group()
