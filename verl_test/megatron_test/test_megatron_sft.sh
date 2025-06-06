set -x
ray stop

export PYTHONPATH=$PYTHONPATH:/data2/zzd/rl_llm/verl

python3 -m megatron_sft_trainer \
    data.train_files=/data2/zzd/data/data/bella_train.parquet \
    data.val_files=/data2/zzd/data/data/bella_train.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=256 \
    data.truncation='right' \
    model.path=/data3/ckpt/Qwen/Qwen2.5-1.5B-Instruct \
    megatron.tensor_model_parallel_size=2 \
    megatron.pipeline_model_parallel_size=1 \
    optim.lr=1e-3 \
    trainer.default_local_dir=/data2/zzd/out_test \
    trainer.project_name=megatron-sft \
    trainer.experiment_name=verl-qwen-megatron-sft \
    trainer.logger=['console'] \
    trainer.total_epochs=2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    use_remove_padding=true