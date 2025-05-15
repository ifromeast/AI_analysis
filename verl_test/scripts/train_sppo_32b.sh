set -x

limo_train_path=/mnt/nvme0/zzd/data/LIMO/train.parquet
limo_test_path=/mnt/nvme0/zzd/data/LIMO/test.parquet
math_train_path=/mnt/nvme0/zzd/data/MATH/train.parquet
math_test_path=/mnt/nvme0/zzd/data/MATH/test.parquet

train_files="['$limo_train_path', '$math_train_path']"
test_files="['$limo_test_path', '$math_test_path']"

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

ray job submit --address="http://10.157.150.10:8265" \
    --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.sppo.main_sppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/mnt/nvme0/zzd/ckpt/Qwen/Qwen2.5-32B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm  \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    custom_reward_function.path=/mnt/nvme0/zzd/verl/math_score.py \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl-h20' \
    trainer.val_before_train=True \
    trainer.experiment_name='sppo_32b_0511' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 $@
    # Note that we set lr_warmup_steps = 15 in config/sppo_trainer.yaml
    # The experiment will converge to 0.656 on MATH dataset after 20 epochs