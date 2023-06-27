import torch
import numpy as np
from collections import defaultdict

# from utils.util import check, get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, args, episode_limit, obs_shape, share_obs_shape, act_shape):
        self.episode_length = episode_limit
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.rnn_hidden_dim
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, args.n_agents, share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, args.n_agents, obs_shape), dtype=np.float32)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, args.n_agents, 1), dtype=np.float32)

        self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, args.n_agents, act_shape), dtype=np.float32)

        self.actions = np.zeros((self.episode_length + 1, self.n_rollout_threads, args.n_agents, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length + 1, self.n_rollout_threads, args.n_agents, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length + 1, self.n_rollout_threads, args.n_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, args.n_agents, 1), dtype=np.float32)

        self.step = 0

    def init_insert(self, share_obs, obs):
        self.share_obs[0] = share_obs.copy()
        self.obs[0] = obs.copy()

    def insert(self, share_obs, obs, actions, action_log_probs,
               value_preds, rewards, masks, available_actions=None):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def last_insert(self, actions, action_log_probs, value_preds):
        self.actions[-1] = actions.copy()
        self.action_log_probs[-1] = action_log_probs.copy()
        self.value_preds[-1] = value_preds.copy()

    def chooseinsert(self, share_obs, obs, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.masks[0] = self.masks[-1].copy()


class ReplayBuffer(object):
    def __init__(self, args, buffer_size, episode_limit, obs_shape, share_obs_shape, act_shape):
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit

        self.episode_limit = episode_limit
        self.rnn_hidden_size = args.rnn_hidden_dim
        self.n_rollout_threads = args.n_rollout_threads
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma

        self.share_obs = np.zeros((self.buffer_size, self.episode_limit, args.n_agents, share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.buffer_size, self.episode_limit, args.n_agents, obs_shape),
                            dtype=np.float32)


        self.available_actions = np.ones((self.buffer_size, self.episode_limit, args.n_agents, act_shape),
                                         dtype=np.float32)

        self.actions = np.zeros((self.buffer_size, self.episode_limit, args.n_agents, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.buffer_size, self.episode_limit, args.n_agents, 1),
                                         dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.episode_limit, args.n_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.buffer_size, self.episode_limit, args.n_agents, 1), dtype=np.float32)


        self.current_size = 0
        self.current_idx = 0

    def insert_episode(self, episode_batch, idx):
        for i in range(self.n_rollout_threads):
            self.share_obs[idx[i], :] = episode_batch.share_obs[:, i]
            self.obs[idx[i], :] = episode_batch.obs[:, i]
            self.available_actions[idx[i], :] = episode_batch.available_actions[:, i]
            self.actions[idx[i], :] = episode_batch.actions[:, i]
            self.rewards[idx[i], :] = episode_batch.rewards[:, i]
            self.masks[idx[i], :] = episode_batch.masks[:, i]


    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.buffer_size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.buffer_size:
            overflow = inc - (self.buffer_size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.buffer_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.buffer_size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer