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

        self.env = env
        # Set the episodic length
        self.episode_limit = self.args.episode_length if args.env == "MPE" else self.args.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        self.log_prefix = "ippo/"

        # Initialize the parameters for history wrapper and environment
        self.max_vehicle_num = args.num_landmarks + args.num_agents + args.num_random_agents
        self.n_agents = args.num_agents
        self.episode_length = args.episode_length

        # Initialize the history wrapper
        self.history_wrapper = observersation_state_history_wrapper(args,
                                                                    self.n_agents,
                                                                    self.max_vehicle_num,
                                                                    self.episode_length,
                                                                    args.max_history_len)

    def setup(self, scheme, groups, preprocess, mac, behavior_learner, prediction_learner):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.behavior_learner = behavior_learner
        self.prediction_learner = prediction_learner

    # Load and unify the parameter name loaded from the environment
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

    # Convert the collective observation into state
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

    # Process the observation of random agent
    def random_obs_process(self, obs):
        n_threads, _, num_entity, obs_dim = obs.shape
        return obs[:, :self.args.n_agents, :, :]

    # Process the reward of random agent
    def random_rwd_process(self, reward):
        reward = reward[:, :self.args.n_agents]
        reward = reward.reshape(self.args.batch_size_run, self.args.n_agents)
        return reward

    # State wrapper for storage in the episodic batch
    def state_wrapper(self, state, envs_not_terminated):
        new_state = np.zeros((self.args.batch_size_run, self.args.state_shape))
        new_state[envs_not_terminated] = state.reshape(self.args.batch_size_run, self.args.state_shape)[envs_not_terminated]
        return new_state

    # Observation wrapper for storage in the episodic batch
    def obs_wrapper(self, obs, envs_not_terminated):
        new_obs = np.zeros((self.args.batch_size_run, self.args.n_agents, self.args.obs_shape))
        new_obs[envs_not_terminated] = obs.reshape(self.args.batch_size_run, self.args.n_agents, self.args.obs_shape)[envs_not_terminated]
        return new_obs

    # Reward wrapper for storage in the episodic batch
    def reward_wrapper(self, reward, envs_not_terminated):
        new_obs = np.zeros((self.args.batch_size_run, self.args.n_agents))
        new_obs[envs_not_terminated] = reward[envs_not_terminated]
        return new_obs

    # Generate the action for random agent (No actual meaning, just place occupation)
    def random_action(self):
        return np.random.randint(self.args.n_actions, size=(self.args.batch_size_run, self.args.num_random_agents))

    # Covert the output actions from the controller into the form recoginizable for environment
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

    def close_env(self):
        self.env.close()

    # Reset environment
    def reset(self):
        self.batch = self.new_batch()

        obs = self.env.reset()

        # Time step for each thread
        self.t = 0

        # Global training time step
        self.env_steps_this_run = 0
        return obs

    def run(self, test_mode=False):
        obs = self.reset()

        # Initialize available actions
        avail_actions = np.ones((self.args.batch_size_run, self.args.n_agents, self.args.n_actions), dtype=np.int64)

        # Initialize metrics for episodes
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        # Process the observation and state from raw observation from the env
        obs = self.random_obs_process(obs)
        state = self.obs2state(obs)

        # Initialize the history wrapper
        self.history_wrapper.agent_obs_profile_init(obs)
        agent_id, obs_vehicle_id, history = self.history_wrapper.obs_history_create(obs)
        single_history_out = self.history_wrapper.obs_single_history_output()

        # Process the observation and state for storage
        state, obs = self.history_wrapper.pure_obs_state_wrapper(state, obs)

        state = self.state_wrapper(state, envs_not_terminated)
        obs = self.obs_wrapper(obs, envs_not_terminated)

        # Initialize the hidden state for controller
        rnn_states_actors = np.zeros((self.args.batch_size_run, self.args.recurrent_N,
                                      self.args.n_agents, self.args.rnn_hidden_dim),
                                      dtype=np.float32)

        rnn_states_critics = np.zeros_like(rnn_states_actors)

        ####################
        # Behavior parameters
        behavior_encoder_rnn = np.zeros((self.args.batch_size_run, self.args.num_encoder_layer, self.args.n_agents,
                                            self.max_vehicle_num, self.args.encoder_rnn_dim), dtype=np.float32)
        behavior_latent = np.zeros((self.args.batch_size_run, self.args.n_agents,
                                    self.max_vehicle_num, self.args.latent_dim), dtype=np.float32)

        ####################
        # Prediction attention parameters
        attention_latent = np.zeros((self.args.batch_size_run, self.args.n_agents,
                                    self.max_vehicle_num, self.args.attention_dim), dtype=np.float32)

        # Update the attention latent
        if self.args.GAT_enable:
            attention_latent = self.prediction_learner.GAT_latent_update \
                (single_history_out, attention_latent, behavior_latent)

        # Store the transition data before the action execution
        pre_transition_data = {
            "state": state,
            "avail_actions": avail_actions,
            "rnn_states_actors": rnn_states_actors,
            "rnn_states_critics": rnn_states_critics,
            "obs": obs,
            "history": single_history_out,
            "behavior_latent": behavior_latent,
            "attention_latent": attention_latent
        }
        self.batch.update(pre_transition_data, ts=0)

        for _ in range(self.args.episode_limit):
            # Initialize the array to store the speed
            speed = np.zeros((self.args.batch_size_run, self.args.n_agents))

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            _, actions, _, rnn_states_actors, rnn_states_critics = self.mac.select_actions_ippo(self.batch,
                                                                                                t_ep=self.t)

            # Convert the action from the controller into actions for execution and storage
            new_actions, action_env = self.action2env_tuple(actions, envs_not_terminated)

            # Update the actions taken
            actions_chosen = {
                "actions": new_actions
            }
            self.batch.update(actions_chosen, ts=self.t, mark_filled=False)

            # Execute the actions
            obs, reward, terminated_agent, env_info = self.env.step(action_env)

            # Process the termination flag and reward
            terminated_agent = terminated_agent[:, :self.args.num_agents]
            reward = self.random_rwd_process(reward)
            reward_all = reward.sum(axis=1)

            terminated = np.array([all(terminated_agent[i, :]) or terminated[i] for i in range(self.batch_size)])

            # Compute the episodic metrics
            for idx in range(self.batch_size):
                episode_returns[idx] += reward_all[idx]
                episode_lengths[idx] += 1 - terminated[idx]

            if not test_mode:
                self.env_steps_this_run += sum(1 - terminated)

            all_terminated = all(terminated)
            if all_terminated:
                break

            # Process the observation and state from raw observation from the env
            obs = self.random_obs_process(obs)
            state = self.obs2state(obs)

            # Get the observation of all opponents
            agent_id, obs_vehicle_id, history = self.history_wrapper.obs_history_create(obs)
            single_history_out = self.history_wrapper.obs_single_history_output()

            # Update the instant incentive
            if self.args.GAT_enable:
                attention_latent = self.prediction_learner.GAT_latent_update\
                    (single_history_out, attention_latent, behavior_latent)

            # Update the behavioral incentive
            if self.args.Behavior_enable:
                # Get the historical observation from the past few steps
                history_out = self.history_wrapper.obs_history_output()
                behavior_latent, behavior_encoder_rnn = \
                    self.behavior_learner.latent_update(history_out, behavior_encoder_rnn, behavior_latent)

            # Process the observation and state for storage
            state, obs = self.history_wrapper.pure_obs_state_wrapper(state, obs)

            state = self.state_wrapper(state, envs_not_terminated)
            obs = self.obs_wrapper(obs, envs_not_terminated)

            post_transition_data = {
                "reward": self.reward_wrapper(reward, envs_not_terminated),
                "terminated": terminated_agent,
                "speed": speed
            }

            pre_transition_data = {
                "state": state,
                "avail_actions": avail_actions,
                "rnn_states_actors": rnn_states_actors,
                "rnn_states_critics": rnn_states_critics,
                "obs": obs,
                "history": single_history_out,
                "behavior_latent": behavior_latent,
                "attention_latent": attention_latent
            }

            # print(pre_transition_data)
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, ts=self.t, mark_filled=True)

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        # Compute the episodic metrics after the whole episode
        avg_rwd = np.mean(episode_returns, axis=0)
        avg_len = np.mean(episode_lengths, axis=0)

        if not test_mode:
            self.t_env += self.env_steps_this_run.tolist()

        self._log(avg_rwd, avg_len)
        self.log_train_stats_t = self.t_env

        return self.batch, None, avg_rwd, avg_len


    def _log(self, episode_reward, episode_len):
        self.logger.log_stat(self.log_prefix + "Average episode_reward", episode_reward, self.t_env)
        self.logger.log_stat(self.log_prefix + "Average episode_len", episode_len, self.t_env)
