#!/bin/sh
# exp param

cd /users/wenshuai/projects/mappo/scripts
# echo $pwd
env="Football"
scenario="academy_corner"
algo="rmappo_opt" # "mappo" "ippo" "rmappo_opt" "mappo_opt"
exp="check"
seed=1

# football param
num_agents=10

# train param
num_env_steps=50000000
episode_length=1000

# echo "python path: ${PYTHONPATH} , ${PYTHONUSERBASE}"
echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 150 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--user_name "wszhao_aalto"
