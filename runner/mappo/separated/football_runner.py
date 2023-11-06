from collections import defaultdict, deque
from itertools import chain
import os
import time
import numpy as np
from functools import reduce
import torch
import wandb
from runner.mappo.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class FootballRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        self.env_infos = defaultdict(list)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * \
                self.episode_length * self.n_rollout_threads
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.env_name,
                              self.algorithm_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                for agent_id in range(self.num_agents):
                    train_infos[agent_id].update({"average_episode_rewards_by_eplength": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                print("average episode rewards of agent 0 is {}".format(train_infos[0]["average_episode_rewards_by_eplength"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps=total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # replay buffer
        # in GRF, we have full observation, so mappo is just ippo
        share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(
            action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(
            rnn_state_critic_collector).transpose(1, 0, 2, 3)
        actions_env = [actions[idx, :, 0]
                       for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=-1)
        if np.any(dones_env):
            for done, info in zip(dones_env, infos):
                if done:
                    self.env_infos["goal"].append(info["score_reward"])
                    if info["score_reward"] > 0:
                        self.env_infos["win_rate"].append(1)
                    else:
                        self.env_infos["win_rate"].append(0)
                    self.env_infos["steps"].append(
                        info["max_steps"] - info["steps_left"])
        # reset rnn and mask args for done envs
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs=share_obs[:, agent_id], 
                                         obs=obs[:, agent_id], 
                                         rnn_states=rnn_states[:, agent_id],
                                         rnn_states_critic=rnn_states_critic[:,agent_id], 
                                         actions=actions[:, agent_id],
                                         action_log_probs=action_log_probs[:, agent_id],
                                         value_preds=values[:, agent_id], 
                                         rewards=rewards[:,agent_id], 
                                         masks=masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads,
                             self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        eval_goal_diffs = np.zeros(self.all_args.eval_episodes)
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(
            self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread !=
                             eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            # get actions
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:,
                                                                      agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            # eval_actions shape will be (100,3,1): batch_size*number_agents*action_dim
            eval_actions_env = [eval_actions[idx, :, 0]
                                for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env)

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goals[num_done] = eval_infos[idx_env]["score_reward"]
                        eval_goal_diffs[num_done] = eval_infos[idx_env]["score"][0] - \
                            eval_infos[idx_env]["score"][1]
                        eval_win_rates[num_done] = 1 if eval_infos[idx_env]["score_reward"] > 0 else 0
                        eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - \
                            eval_infos[idx_env]["steps_left"]
                        # print("episode {:>2d} done by env {:>2d}: {}".format(num_done, idx_env, eval_infos[idx_env]["score_reward"]))
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread !=
                                 eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        # get expected goal
        eval_goal = np.mean(eval_goals)
        eval_goal_diff = np.mean(eval_goal_diffs)
        eval_win_rate = np.mean(eval_win_rates)
        eval_step = np.mean(eval_steps)

        # log and print
        print("eval expected goal is {}.".format(eval_goal))
        if self.use_wandb:
            wandb.log({"eval_goal": eval_goal}, step=total_num_steps)
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_goal_diff": eval_goal_diff}, step=total_num_steps)
            wandb.log({"eval_step": eval_step}, step=total_num_steps)
        if self.use_writter:
            self.writter.add_scalars(
                "eval_goal", {"expected_goal": eval_goal}, total_num_steps)
            self.writter.add_scalars(
                "eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
            self.writter.add_scalars(
                "eval_goal_diff", {"eval_goal_diff": eval_goal_diff}, total_num_steps)
            self.writter.add_scalars(
                "eval_step", {"expected_step": eval_step}, total_num_steps)

    @torch.no_grad()
    def render(self):
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[
                    0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
                # self.trainer.prep_rollout()
                # render_actions, render_rnn_states = self.trainer.policy.act(
                #     np.concatenate(render_obs),
                #     np.concatenate(render_rnn_states),
                #     np.concatenate(render_masks),
                #     deterministic=True
                # )

                # # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                # render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                # render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                # render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                eval_actions_collector = []
                eval_rnn_states_collector = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(render_obs[:, agent_id],
                                                          render_rnn_states[:,
                                                                            agent_id],
                                                          render_masks[:,
                                                                       agent_id],
                                                          deterministic=True)
                    render_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                    render_actions = np.array(
                        eval_actions_collector).transpose(1, 0, 2)
                    # eval_actions shape will be (100,3,1): batch_size*number_agents*action_dim
                    render_actions_env = [render_actions[idx, :, 0]
                                          for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(
                    render_actions_env)

                # append frame
                if self.all_args.save_gifs:
                    image = render_infos[0]["frame"]
                    frames.append(image)

            # print goal
            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(
                i_episode, render_rewards[0, 0]))

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir),
                                                  i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )

        print("expected goal: {}".format(np.mean(render_goals)))
