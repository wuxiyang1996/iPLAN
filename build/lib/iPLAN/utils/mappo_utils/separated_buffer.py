import torch as th
import numpy as np
from collections import deque


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_shape, share_obs_shape 
                 ):
        self.episode_limit = args.episode_limit   # max ep len
        self.buffer_size = args.buffer_size  # originally n_rollout_threads
        self.thread_id = 0
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.recurrent_N = args.recurrent_N
        self.batch_size_run = args.batch_size_run

        self.state = deque(maxlen=self.buffer_size)
        self.obs = deque(maxlen=self.buffer_size)
        self.rnn_states_actor = deque(maxlen=self.buffer_size)
        self.rnn_states_critic = deque(maxlen=self.buffer_size)
        self.actions = deque(maxlen=self.buffer_size)
        self.actions_onehot = deque(maxlen=self.buffer_size)
        self.reward = deque(maxlen=self.buffer_size)
        self.terminated_masks = deque(maxlen=self.buffer_size)
        self.active_masks = deque(maxlen=self.buffer_size)
        self.available_actions = deque(maxlen=self.buffer_size)

        self.histories = deque(maxlen=self.buffer_size)
        self.behavior_latent = deque(maxlen=self.buffer_size)

        self.attention_latent = deque(maxlen=self.buffer_size)

    def can_sample(self):
        # print("LEN SELF OBS IS ", len(self.obs))
        # print("LEN BUFFER SIZE IS ", self.buffer_size)
        return len(self.obs) == self.buffer_size

    def data_wrapper(self, element, idx):
        return element[idx].reshape(1, *element.shape[1:])

    def insert(self, state, obs, rnn_states_actor, rnn_states_critic, 
               actions, actions_onehot, rewards, terminated_masks,
               histories, behavior_latent, attention_latent, active_masks=None, available_actions=None):

        for i in range(self.batch_size_run):
            self.state.append(self.data_wrapper(state, i))
            self.obs.append(self.data_wrapper(obs, i))
            self.rnn_states_actor.append(self.data_wrapper(rnn_states_actor, i))
            self.rnn_states_critic.append(self.data_wrapper(rnn_states_critic, i))
            self.actions.append(self.data_wrapper(actions, i))
            self.actions_onehot.append(self.data_wrapper(actions_onehot, i))
            self.reward.append(self.data_wrapper(rewards, i))
            self.terminated_masks.append(self.data_wrapper(terminated_masks, i))

            self.histories.append(self.data_wrapper(histories, i))
            self.behavior_latent.append(self.data_wrapper(behavior_latent, i))

            self.attention_latent.append(self.data_wrapper(attention_latent, i))

            if active_masks is not None:
                self.active_masks.append(self.data_wrapper(active_masks, i))
            if available_actions is not None:
                self.available_actions.append(self.data_wrapper(available_actions, i))

    def get_batch(self):
        if len(self.obs) < self.buffer_size:
            return
        batch = {}
        batch['obs'] = th.cat(list(self.obs))
        batch['state'] = th.cat(list(self.state))
        batch['actions'] = th.cat(list(self.actions))
        batch['actions_onehot'] = th.cat(list(self.actions_onehot))
        batch['rnn_states_actor'] = th.cat(list(self.rnn_states_actor))
        batch['rnn_states_critic'] = th.cat(list(self.rnn_states_critic))
        batch['reward'] = th.cat(list(self.reward))
        batch['terminated_masks'] = th.cat(list(self.terminated_masks))

        batch['history'] = th.cat(list(self.histories))
        batch['behavior_latent'] = th.cat(list(self.behavior_latent))

        batch['attention_latent'] = th.cat(list(self.attention_latent))

        if len(self.active_masks) != 0:
            batch['active_masks'] = th.cat(list(self.active_masks))
        if len(self.available_actions) != 0:
            batch['available_actions'] = th.cat(list(self.available_actions))

        return batch

    def clear_buffer(self):
        self.state.clear()
        self.obs.clear()
        self.rnn_states_actor.clear()
        self.rnn_states_critic.clear()
        self.actions.clear()
        self.actions_onehot.clear()
        self.reward.clear()
        self.terminated_masks.clear()
        self.available_actions.clear()

        self.histories.clear()
        self.behavior_latent.clear()

        self.attention_latent.clear()
