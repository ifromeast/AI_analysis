
import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import ray
from functools import partial
from contextlib import nullcontext
from omegaconf import OmegaConf, open_dict
from codetiming import Timer
from typing import Dict, Iterable, Optional, Type
from pprint import pprint

import hydra
import torch
import torch.distributed
# from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
# from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
# from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from verl import DataProto
import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.torch_dtypes import PrecisionType
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches, masked_mean
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup


from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.utils.model import load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits


from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


# def extract_step(path):
#     match = re.search(r"global_step_(\d+)", path)
#     if match:
#         return int(match.group(1))
#     return None


# def convert_to_regular_types(obj):
#     """Convert Hydra configs and other special types to regular Python types."""
#     from omegaconf import DictConfig, ListConfig

#     if isinstance(obj, (ListConfig, DictConfig)):
#         return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
#     elif isinstance(obj, (list, tuple)):
#         return [convert_to_regular_types(x) for x in obj]
#     elif isinstance(obj, dict):
#         return {k: convert_to_regular_types(v) for k, v in obj.items()}
#     return obj


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


# @ray.remote
class MegatronSFTWorker(MegatronWorker):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        self._normalize_config_bsz()

        # self.loss_fct = nn.CrossEntropyLoss(reduction="none")


    def _normalize_config_bsz(self):
        dp_size = mpu.get_data_parallel_world_size()
        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

        self._is_offload_param = self.config.megatron.get("param_offload", False)
        self._is_offload_grad = self.config.megatron.get("grad_offload", False)
        self._is_offload_optimizer = self.config.megatron.get("optimizer_offload", False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        optim_config = self.config.optim
        self.actor_module, self.actor_optimizer, self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during init", logger=None)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during init", logger=None)

        # self.actor = MegatronPPOActor(
        #     config=self.config,
        #     model_config=self.actor_model_config,
        #     hf_config=self.hf_config,
        #     tf_config=self.tf_config,
        #     actor_module=self.actor_module,
        #     actor_optimizer=self.actor_optimizer,
        # )
        # log_gpu_memory_usage("After MegatronPPOActor init", logger=None)

        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            model_config=self.actor_model_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            tokenizer=self.tokenizer,
            optimizer=self.actor_optimizer,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    def _build_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config)
        self.generation_config = get_generation_config(self.local_path)

        def megatron_actor_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, share_embeddings_and_output_weights=self.share_embeddings_and_output_weights, value=False, fix_moe_router=override_model_config.get("moe_config", {}).get("fix_moe_router", False))
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        actor_module = get_model(
            megatron_actor_model_provider,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        print(f"actor_module: {len(actor_module)}")

        if self.config.megatron.use_dist_checkpointing:
            load_mcore_dist_weights(actor_module, self.config.megatron.dist_checkpointing_path, is_value_model=False)
        else:
            load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)

        if self.rank == 0:
            print_model_size(actor_module[0])
        log_gpu_memory_usage("After MegatronPPOActor init", logger=None)


        # TODO: add more optimizer args into config
        optim_config = init_megatron_optim_config(optim_config)
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        log_gpu_memory_usage("After actor optimizer init", logger=None)

        return actor_module, actor_optimizer, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: DataProto):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=None)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=None)
        data.batch = data.batch.cuda()

        micro_batch_size = self.config.data.micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.update_policy(dataloader=dataloader)
        delta_time = timer.last
        # global_num_tokens = data.meta_info["global_token_num"]
        # estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        # metrics["perf/mfu/actor"] = estimated_flops * self.config.trainer.total_epochs / promised_flops / self.world_size

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=None)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=None)

        torch.cuda.empty_cache()
        return output 

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor
        Args:
            data (DataProto): a DataProto containing keys
                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where ``sequence_length = prompt_length + response_length``
                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64
                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64
                ``loss_mask``: tensor of shape [batch_size, sequence_length]. torch.int64
        Returns:
        """
        select_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.data.train_batch_size,
            epochs=self.config.trainer.total_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.data.shuffle},
        )
    
    @GPUMemoryLogger(role="megatron actor", logger=None)
    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        # self.prof.start()
        for data in dataloader:
            # data = data.batch.to(self.actor_module.device)
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            metric_micro_batch = self.forward_backward_batch(data)
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            learning_rate = self.actor_optimizer.param_groups[-1]["lr"]
            data = {"actor/grad_norm": grad_norm, "actor/lr": learning_rate}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            # self.prof.step()
        # add empty cache after each compute
        # self.prof.stop_and_save()
        # self.prof.stop_trace()
        torch.cuda.empty_cache()
        return metrics
    


    def forward_backward_batch(self, data: DataProto, forward_only=False):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        broadcast_dict_tensor(data.batch, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)

        if data.meta_info.get("micro_batch_size", None) is not None:
            batch_size = data.meta_info["micro_batch_size"]
        else:
            batch_size = self.config.micro_batch_size_per_gpu
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
        # compute input shapes for pp stages
        n_micro_batch = len(batches)
        seq_len = batches[0]["input_ids"].shape[1]

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))
        forward_backward_func = get_forward_backward_func()

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            label_mask = batch.pop("loss_mask")[:, 1:].bool()  # remove the first token, which is the prompt token

            labels = input_ids[:, 1:].contiguous()

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape

                ret = {}

                entropy = vocab_parallel_entropy(logits)
                ret["entropy"] = entropy

                log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                log_probs = log_probs.masked_fill(~label_mask, 0.0)
                ret["log_probs"] = log_probs
                # ret["loss"] = -log_probs.sum(dim=-1)  # sum over vocab dimension
                return ret

            logits_processor_args = {"label": labels, "label_mask": label_mask}

            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)
            print("[0]input_ids shape: ", input_ids.shape)
            print("[0]attention_mask shape: ", attention_mask.shape)
            print("position_ids shape: ", position_ids.shape)
            print("labels shape: ", labels.shape)
            print("label_mask shape: ", label_mask.shape)
            # output = forward_fn(model, input_ids[:,:-1], attention_mask[:,:-1], position_ids[:,:-1], sequence_parallel=self.tf_config.sequence_parallel, pack_seqs=False)

            output = forward_fn(model, input_ids[:,:-1], attention_mask[:,:-1], position_ids[:,:-1], sequence_parallel=self.tf_config.sequence_parallel,logits_processor=logits_processor, logits_processor_args=logits_processor_args)

            def loss_func(output):
                """
                Compute the loss from the output and label.
                """
                metrics = {}
                log_probs = output["log_probs"]
                
                loss = -log_probs.sum(dim=-1)  # sum over vocab dimension
                metrics["loss"] = loss.detach().item()
                return loss, [metrics, output["entropy"]]
            # print("log_probs shape: ", output["log_probs"].shape)
            return output, partial(loss_func)

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        return losses_reduced



class RaySFTTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[str, MegatronSFTWorker],
        resource_pool: RayResourcePool,
        ray_worker_group: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        assert "megatron_worker" in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.worker_mapping = role_worker_mapping
        self.resource_pool = resource_pool
        self.ray_worker_group = ray_worker_group   
        self.device_name = device_name

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)


    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for megatron sft worker
        """
        megatron_worker = RayClassWithInitArgs(cls=self.worker_mapping["megatron_worker"], config=self.config, tokenizer=self.tokenizer)

        all_wg = {}
        class_dict = {"megatron_worker": megatron_worker}
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = self.ray_worker_group(resource_pool=self.resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

        self.megatron_worker_group = all_wg["megatron_worker"]
        self.megatron_worker_group.init_model()
        
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        # self._load_checkpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                is_last_step = self.global_steps >= self.total_training_steps
                output = self.megatron_worker_group.update_actor(batch)



@hydra.main(config_path=".", config_name="megatron_sft", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN",
             "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


def create_sft_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer
        # 

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # megatron_worker = MegatronSFTWorker(config, tokenizer)
        worker_mapping = {"megatron_worker": ray.remote(MegatronSFTWorker)}
        global_pool_id = "global_pool"
        process_on_nodes = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=global_pool_id)
        # mapping = {"megatron_worker": global_pool_id}

        ray_worker_group = NVMegatronRayWorkerGroup
        # resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
        val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
        train_sampler = create_sft_sampler(config.data, train_dataset)
        pprint(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        trainer = RaySFTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=worker_mapping,
            resource_pool=resource_pool,
            ray_worker_group=ray_worker_group,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=None,  # Use default collate_fn
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        trainer.init_workers()
        trainer.fit()

if __name__ == "__main__":
    main()
