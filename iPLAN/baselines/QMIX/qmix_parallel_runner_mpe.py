from functools import partial
import torch
from components.episode_buffer import EpisodeBatch
import numpy as np
from observation_wrapper import observersation_state_history_wrapper
from utils.mappo_utils.util import get_shape_from_act_space, get_shape_from_obs_space
import time
from PIL import Image

class ParallelRunner:

    def __init__(self, args, env, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # assert self.batch_size == 1

        self.env = env
        self.episode_limit = self.args.episode_length if args.env == "MPE" else self.args.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        self.max_vehicle_num = args.num_landmarks + args.num_agents + args.num_random_agents
        self.n_agents = args.num_agents
        self.episode_length = args.episode_length

        self.history_wrapper = observersation_state_history_wrapper(args,
                                                                    self.n_agents,
                                                                    self.max_vehicle_num,
                                                                    self.episode_length,
                                                                    None)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess


    def get_env_info(self, args):
        obs_shape = args.obs_shape_single * self.max_vehicle_num

        env_info = {
            "n_agents": self.n_agents,
            "n_actions": args.n_actions,
            "state_shape": args.obs_shape_single * self.max_vehicle_num,
            "episode_limit": self.episode_length,
            "obs_shape": obs_shape,
        }
        return env_info

    def obs2state(self, obs):
        n_threads, n_agents, num_entity, obs_dim = obs.shape
        state = np.zeros((n_threads, num_entity, obs_dim))
        for i in range(n_threads):
            agent_pos = obs[i, 0, 0, 1:3]
            state[i, 0, :] = obs[i, 0, 0, :]
            for j in range(1, num_entity):
                state[i, j, 0] = obs[i, 0, j, 0]
                state[i, j, 1:3] = obs[i, 0, j, 1:3] + agent_pos
                state[i, j, 3:5] = obs[i, 0, j, 3:5]
        return state.reshape((n_threads, 1, num_entity * obs_dim))

    def random_obs_process(self, obs):
        n_threads, _, num_entity, obs_dim = obs.shape
        return obs[:, :self.args.n_agents, :, :]

    def random_rwd_process(self, reward):
        reward = reward[:, :self.args.n_agents]
        reward = reward.reshape(self.args.batch_size_run, self.args.n_agents)
        return reward

    def state_wrapper(self, state, envs_not_terminated):
        new_state = np.zeros((self.args.batch_size_run, self.args.state_shape))
        new_state[envs_not_terminated] = state.reshape(self.args.batch_size_run, self.args.state_shape)[envs_not_terminated]
        return new_state

    def obs_wrapper(self, obs, envs_not_terminated):
        new_obs = np.zeros((self.args.batch_size_run, self.args.n_agents, self.args.obs_shape))
        new_obs[envs_not_terminated] = obs.reshape(self.args.batch_size_run, self.args.n_agents, self.args.obs_shape)[envs_not_terminated]
        return new_obs

    def reward_wrapper(self, reward, envs_not_terminated):
        new_obs = np.zeros((self.args.batch_size_run, ))
        new_obs[envs_not_terminated] = reward[envs_not_terminated]
        return new_obs

    def random_action(self):
        return np.random.randint(self.args.n_actions, size=(self.args.batch_size_run, self.args.num_random_agents))

    def action2env_tuple(self, actions, envs_not_terminated):
        new_actions = np.zeros((self.args.batch_size_run, self.args.n_agents))
        new_actions[envs_not_terminated] = actions[envs_not_terminated]

        if self.args.num_random_agents > 0:
            rand_action = self.random_action()

        action_env = np.zeros((self.args.batch_size_run, self.args.n_agents + self.args.num_random_agents, self.args.n_actions))
        for i in range(self.args.batch_size_run):
            for j in range(self.args.n_agents):
                action_env[i, j, int(new_actions[i, j])] = 1

        if self.args.num_random_agents > 0:
            for i in range(self.args.batch_size_run):
                for j in range(self.args.num_random_agents):
                    action_env[i, self.args.n_agents + j, int(rand_action[i, j])] = 1
        return new_actions, action_env

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()

        obs = self.env.reset()
        self.t = 0
        self.env_steps_this_run = 0

        return obs

    def run(self, test_mode=False):
        obs = self.reset()

        avail_actions = np.ones((self.args.batch_size_run, self.args.n_agents, self.args.n_actions), dtype=np.int64)

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        obs = self.random_obs_process(obs)
        state = self.obs2state(obs)

        self.history_wrapper.agent_obs_profile_init(obs)
        state, obs = self.history_wrapper.pure_obs_state_wrapper(state, obs)
        state = self.state_wrapper(state, envs_not_terminated)
        obs = self.obs_wrapper(obs, envs_not_terminated)

        pre_transition_data = {
            "state": state,
            "avail_actions": avail_actions,
            "obs": obs
        }
        self.batch.update(pre_transition_data, ts=0)

        for _ in range(self.args.episode_limit):
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            speed = np.zeros((self.args.batch_size_run, self.args.n_agents))
            new_actions, action_env = self.action2env_tuple(cpu_actions, envs_not_terminated)

            # Update the actions taken
            actions_chosen = {
                "actions": new_actions
            }
            self.batch.update(actions_chosen, ts=self.t, mark_filled=False)

            obs, reward, terminated_agent, env_info = self.env.step(action_env)
            terminated_agent = terminated_agent[:, :self.args.num_agents]
            reward = self.random_rwd_process(reward)
            reward = reward.sum(axis=1)

            if self.args.animation_enable:
                img = self.env.render("rgb_array")

                for i in range(self.batch_size):
                    im = Image.fromarray(img[i])
                    im.save("animation/" + str(i) + "/" + str(self.t) + ".jpg")

            terminated = np.array([all(terminated_agent[i, :]) or terminated[i] for i in range(self.batch_size)])

            for idx in range(self.batch_size):
                episode_returns[idx] += reward[idx]
                episode_lengths[idx] += 1 - terminated[idx]

            if not test_mode:
                self.env_steps_this_run += sum(1 - terminated)

            all_terminated = all(terminated)
            if all_terminated:
                break

            obs = self.random_obs_process(obs)
            state = self.obs2state(obs)

            state, obs = self.history_wrapper.pure_obs_state_wrapper(state, obs)

            state = self.state_wrapper(state, envs_not_terminated)
            obs = self.obs_wrapper(obs, envs_not_terminated)

            post_transition_data = {
                "reward": self.reward_wrapper(reward, envs_not_terminated),
                "agent_terminated": terminated_agent,
                "terminated": terminated,
                "speed": speed
            }

            pre_transition_data = {
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs
            }

            # print(pre_transition_data)
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, ts=self.t, mark_filled=True)

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]


        avg_rwd = np.mean(episode_returns, axis=0)
        avg_len = np.mean(episode_lengths, axis=0)

        if not test_mode:
            self.t_env += self.env_steps_this_run.tolist()

        if self.t_env - self.log_train_stats_t >= self.args.log_interval:
            self._log(avg_rwd, avg_len)
            self.log_train_stats_t = self.t_env

        return self.batch, None, avg_rwd, avg_len

    def _log(self, episode_reward, episode_len):
        self.logger.log_stat(self.args.log_prefix + "Average episode_reward", episode_reward, self.t_env)
        self.logger.log_stat(self.args.log_prefix + "Average episode_len", episode_len, self.t_env)

