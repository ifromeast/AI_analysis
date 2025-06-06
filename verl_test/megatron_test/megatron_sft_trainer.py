
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

import hydra
import torch
import torch.distributed
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm


from verl import DataProto
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
from verl.utils.metric import reduce_metrics

from megatron_sft_worker import MegatronSFTWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))




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
                output_metrics = reduce_metrics(output.meta_info["metrics"])
                metrics.update(output_metrics)

                metrics.update({"global_step": self.global_steps, "epoch": epoch,})
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    # pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


@hydra.main(config_path=".", config_name="megatron_sft", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
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

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        worker_mapping = {"megatron_worker": ray.remote(MegatronSFTWorker)}
        global_pool_id = "global_pool"
        process_on_nodes = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=global_pool_id)
        ray_worker_group = NVMegatronRayWorkerGroup

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
