#!/bin/sh

python ./train/train_mujoco_1.py --use_recurrent_policy --num_mini_batch 40 --env_name "mujoco" --scenario_name "HalfCheetah-v2" --agent_conf "6x1"\
                                    --agent_obsk 2 --lr 0.00005 --critic_lr 0.005 --n_training_thread 32 --n_rollout_threads 32 --num_env_steps 40000000 \
                                    --episode_length 1000 --ppo_epoch 5 --algorithm_name "mappo_opt" --use_recurrent_policy \
                                    --use_eval --eval_interval 10 --n_eval_rollout_threads 1 --share_policy
