#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket
sys.path.append("../")

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb

# code repository sub-packages
from configs.matrix_config import get_config
from envs.matrix.matrix_env import ClimbingEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from datetime import datetime

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "matrix":
                env = ClimbingEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "matrix":
                env = ClimbingEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="climbing", 
                        help="one of climbing, penalty_100, penalty_75, penalty_50, penalty_25, penalty_0 ")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="number of controlled players.")
    parser.add_argument("--eval_deterministic", action="store_false", 
                        default=True, 
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")

    parser.add_argument("--save_videos", action="store_true", default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "rmappo_opt":
        print("u are choosing to use rmappo_opt, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        all_args.use_nonnegtive_opt = True
    elif all_args.algorithm_name == "mappo_opt":
        print("u are choosing to use mappo_opt, we set use_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        all_args.use_nonnegtive_opt = True
    elif all_args.algorithm_name == "rmappo_clip":
        print("u are choosing to use rmappo_clip, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        all_args.use_nonnegtive_opt = False
        all_args.use_opt_clip=True
        all_args.use_adv_nml=False
    elif all_args.algorithm_name == "mappo_clip":
        print("u are choosing to use mappo_clip, we set use_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        all_args.use_nonnegtive_opt = False
        all_args.use_opt_clip=True
        all_args.use_adv_nml=False
    else:
        raise NotImplementedError

    # seed
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.seed
    else:
        all_args.seed=np.random.randint(1000,10000)
    print("seed is :",all_args.seed)
    
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    cur_dir=Path(os.path.join(os.path.dirname(__file__)))
    result_dir=os.path.join(cur_dir.parent.absolute().parent.absolute(), "results")
    run_dir = Path(result_dir) / all_args.env_name / all_args.scenario_name / all_args.algorithm_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    date_time=datetime.now().strftime("%Y%m%d%H%M%S")

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name="-".join([
                            all_args.algorithm_name,
                            "seed" + str(all_args.seed)
                         ]),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         tags=[all_args.wandb_tag],
                         reinit=True)
    if all_args.use_writter:
        w_run_dir = run_dir / 'tb'/ Path('run_'+str(all_args.seed)+'_'+date_time)
        if not w_run_dir.exists():
            os.makedirs(str(w_run_dir))
    else:
        w_run_dir=run_dir

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.scenario_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": w_run_dir
    }

    # run experiments
    assert all_args.share_policy == False, "only separated policy is supported"
    if not all_args.share_policy:
        from runner.mappo.separated.matrix_runner import MatrixRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    if all_args.use_writter:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
