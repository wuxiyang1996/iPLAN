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

        # Initialize the parameters for history wrapper and environment
        self.max_vehicle_num = args.n_other_vehicles + args.n_agents
        self.n_agents = args.n_agents
        self.episode_length = args.episode_limit

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
        obs_shape = args.obs_shape_single * args.n_obs_vehicles
        env_info = {
            "n_agents": self.n_agents,
            "n_actions": args.n_actions,
            "state_shape": args.obs_shape_single * self.max_vehicle_num,
            "episode_limit": self.episode_length,
            "obs_shape": obs_shape,
        }
        return env_info

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

    # Covert the output actions from the controller into the form recoginizable for environment
    def action2env_tuple(self, actions, envs_not_terminated):
        new_actions = np.zeros((self.args.batch_size_run, self.args.n_actions))
        new_actions[envs_not_terminated] = actions[envs_not_terminated]
        action_env = []
        for i in range(self.args.batch_size_run):
            action_env.append(tuple(new_actions[i, :]))
        return new_actions, action_env

    def close_env(self):
        self.env.close()

    # Reset environment
    def reset(self):
        self.batch = self.new_batch()

        state, obs = self.env.reset()

        # Time step for each thread
        self.t = 0

        # Global training time step
        self.env_steps_this_run = 0
        return state, obs

    def run(self, test_mode=False):
        state, obs = self.reset()

        # Initialize available actions
        avail_actions = np.ones((self.args.batch_size_run, self.args.n_agents, self.args.n_actions), dtype=np.int64)

        # Initialize metrics for episodes
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_wins = [0 for _ in range(self.batch_size)]
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

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
            state, obs, reward, win_tags, terminated_agent, env_info = self.env.step(action_env)

            # Compute the sum of reward for visualization
            reward_all = reward.sum(axis=1)

            # Get the screenshot for animation
            if self.args.animation_enable:
                img = self.env.render("rgb_array")

                for i in range(self.batch_size):
                    im = Image.fromarray(img[i])
                    im.save("animation/" + str(i) + "/" + str(self.t) + ".jpg")

            if self.args.env == "highway":
                for i in range(self.batch_size):
                    speed[i, :] = env_info[i]["speed"]

            terminated = np.array([all(terminated_agent[i, :]) or terminated[i] for i in range(self.batch_size)])

            # Compute the episodic metrics
            for idx in range(self.batch_size):
                episode_returns[idx] += reward_all[idx]
                episode_lengths[idx] += 1 - terminated[idx]
                episode_wins[idx] = sum(win_tags[idx, :])

            if not test_mode:
                self.env_steps_this_run += sum(1 - terminated)

            all_terminated = all(terminated)
            if all_terminated:
                break

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

            # Store the transition data after the action execution
            post_transition_data = {
                "reward": self.reward_wrapper(reward, envs_not_terminated),
                "terminated": terminated_agent,
                "speed": speed
            }

            # Store the transition data before the next action execution
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


            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, ts=self.t, mark_filled=True)

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        # Compute the episodic metrics after the whole episode
        avg_win_rates = np.mean(episode_wins, axis=0)
        avg_rwd = np.mean(episode_returns, axis=0)
        avg_len = np.mean(episode_lengths, axis=0)

        if not test_mode:
            self.t_env += self.env_steps_this_run.tolist()

        self._log(avg_win_rates, avg_rwd, avg_len)
        self.log_train_stats_t = self.t_env

        return self.batch, avg_win_rates, avg_rwd, avg_len

    # Store the log values into the tensorboard
    def _log(self, win_rates, episode_reward, episode_len):
        self.logger.log_stat(self.args.log_prefix + "Average episode_win_num", win_rates, self.t_env)
        self.logger.log_stat(self.args.log_prefix + "Average episode_reward", episode_reward, self.t_env)
        self.logger.log_stat(self.args.log_prefix + "Average episode_len", episode_len, self.t_env)
