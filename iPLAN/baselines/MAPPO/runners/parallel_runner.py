from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
from observation_wrapper import observersation_state_history_wrapper
from PIL import Image

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, env, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.env = env
        self.episode_limit = self.args.episode_length if args.env == "MPE" else self.args.episode_limit

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        self.max_vehicle_num = args.n_other_vehicles + args.n_agents
        self.n_agents = args.n_agents
        self.episode_length = args.episode_limit

        self.history_wrapper = observersation_state_history_wrapper(args,
                                                                    self.n_agents,
                                                                    self.max_vehicle_num,
                                                                    self.episode_length,
                                                                    args.max_history_len)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self, args):
        obs_shape = args.obs_shape_single * args.n_obs_vehicles

        env_info = {
            "n_agents": self.n_agents,
            "n_actions": args.n_actions,
            "state_shape": args.obs_shape_single * self.max_vehicle_num,
            "episode_limit": self.episode_length,
            "obs_shape": obs_shape,
        }
        return env_info

    def state_wrapper(self, state, envs_not_terminated):
        new_state = np.zeros((self.args.batch_size_run, self.args.state_shape))
        new_state[envs_not_terminated] = state.reshape(self.args.batch_size_run, self.args.state_shape)[envs_not_terminated]
        return new_state

    def obs_wrapper(self, obs, envs_not_terminated):
        new_obs = np.zeros((self.args.batch_size_run, self.args.n_agents, self.args.obs_shape))
        new_obs[envs_not_terminated] = obs.reshape(self.args.batch_size_run, self.args.n_agents, self.args.obs_shape)[envs_not_terminated]
        return new_obs

    def reward_wrapper(self, reward, envs_not_terminated):
        new_obs = np.zeros((self.args.batch_size_run, 1, 1))
        new_obs[envs_not_terminated, 0, 0] = reward[envs_not_terminated]
        return new_obs

    def action2env_tuple(self, actions, envs_not_terminated):
        new_actions = np.zeros((self.args.batch_size_run, self.args.n_actions))
        new_actions[envs_not_terminated] = actions[envs_not_terminated]
        action_env = []
        for i in range(self.args.batch_size_run):
            action_env.append(tuple(new_actions[i, :]))
        return new_actions, action_env

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()

        state, obs = self.env.reset()
        self.t = 0
        self.env_steps_this_run = 0

        return state, obs

    def run(self, test_mode=False):
        state, obs = self.reset()
        self.mac.init_hidden(self.args.batch_size_run)
        avail_actions = np.ones((self.args.batch_size_run, self.args.n_agents, self.args.n_actions), dtype=np.int64)

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_wins = [0 for _ in range(self.batch_size)]
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        if self.args.env == "MPE":
            obs = self.random_obs_process(obs)
            state = self.obs2state(obs)

        self.history_wrapper.agent_obs_profile_init(obs)
        state, obs = self.history_wrapper.pure_obs_state_wrapper(state, obs)

        state = self.state_wrapper(state, envs_not_terminated)
        obs = self.obs_wrapper(obs, envs_not_terminated)

        pre_transition_data = {
            "state": state,
            "avail_actions": avail_actions,
            "obs": obs,
        }
        self.batch.update(pre_transition_data, ts=0)

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            speed = np.zeros((self.args.batch_size_run, self.args.n_agents))

            new_actions, action_env = self.action2env_tuple(cpu_actions, envs_not_terminated)

            # Update the actions taken
            actions_chosen = {
                "actions": new_actions
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            state, obs, reward, win_tags, terminated_agent, env_info = self.env.step(action_env)


            reward_all = reward.sum(axis=1)

            if self.args.animation_enable:
                img = self.env.render("rgb_array")

                for i in range(self.batch_size):
                    im = Image.fromarray(img[i])
                    im.save("animation/" + str(i) + "/" + str(self.t) + ".jpg")

            for i in range(self.batch_size):
                speed[i, :] = env_info[i]["speed"]

            terminated = np.array([all(terminated_agent[i, :]) or terminated[i] for i in range(self.batch_size)])

            for idx in range(self.batch_size):
                episode_returns[idx] += reward_all[idx]
                episode_lengths[idx] += 1 - terminated[idx]
                episode_wins[idx] = sum(win_tags[idx, :])

            if not test_mode:
                self.env_steps_this_run += sum(1 - terminated)

            all_terminated = all(terminated)
            if all_terminated:
                break

            state, obs = self.history_wrapper.pure_obs_state_wrapper(state, obs)

            state = self.state_wrapper(state, envs_not_terminated)
            obs = self.obs_wrapper(obs, envs_not_terminated)

            post_transition_data = {
                "reward": self.reward_wrapper(reward_all, envs_not_terminated),
                "agent_terminated": terminated_agent,
                "terminated": terminated,
                "speed": speed
            }

            pre_transition_data = {
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
            }

            # print(pre_transition_data)
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        avg_win_rates = np.mean(episode_wins, axis=0)
        avg_rwd = np.mean(episode_returns, axis=0)
        avg_len = np.mean(episode_lengths, axis=0)

        if not test_mode:
            self.t_env += self.env_steps_this_run.tolist()

        if self.t_env - self.log_train_stats_t >= self.args.log_interval:
            self._log(avg_win_rates, avg_rwd, avg_len)
            self.log_train_stats_t = self.t_env

        return self.batch, avg_win_rates, avg_rwd, avg_len

    def _log(self, win_rates, episode_reward, episode_len):
        self.logger.log_stat(self.args.log_prefix + "Average episode_win_num", win_rates, self.t_env)
        self.logger.log_stat(self.args.log_prefix + "Average episode_reward", episode_reward, self.t_env)
        self.logger.log_stat(self.args.log_prefix + "Average episode_len", episode_len, self.t_env)


