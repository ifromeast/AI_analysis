set -x
ray stop
# export LOCAL_RANK=0
# export RANK=0
# export WORLD_SIZE=8
# export MASTER_ADDR=localhost
# export MASTER_PORT=12345

python3 -m megatron_sft_trainer \
    data.train_files=/data2/zzd/data/data/bella_train.parquet \
    data.val_files=/data2/zzd/data/data/bella_train.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    optim.lr=1e-6 \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=256 \
    data.truncation='right' \
    model.path=/data3/ckpt/Qwen/Qwen2.5-0.5B-Instruct \
    megatron.tensor_model_parallel_size=2 \
    megatron.pipeline_model_parallel_size=4 \
    trainer.default_local_dir=/data2/zzd/out_test \
    trainer.project_name=megatron-sft \
    trainer.experiment_name=verl-qwen-gsm8k-sft \
    trainer.logger=['console'] \
    trainer.total_epochs=2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    use_remove_padding=true