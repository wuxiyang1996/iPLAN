import torch as th
import torch.nn as nn

from utils.mappo_utils.mlp import MLPBase
from utils.mappo_utils.rnn import RNNLayer
from utils.mappo_utils.act import ACTLayer
from utils.mappo_utils.util import check


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (th.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, input_shape, args):
        super(R_Actor, self).__init__()
        if args.use_cuda:
            self.device = th.device("cuda")
        else:
            self.device = th.device("cpu")

        self.rnn_hidden_dim = args.rnn_hidden_dim
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=th.float32, device=self.device)

        self.base = MLPBase(args, input_shape)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.rnn_hidden_dim, self.rnn_hidden_dim,
                                self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(args.n_actions, self.rnn_hidden_dim,
                            self._use_orthogonal, self._gain)
        self.to(self.device)

    def forward(self, obs, rnn_states, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / th.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / th.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / th.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / th.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (th.Tensor) actions to take.
        :return action_log_probs: (th.Tensor) log probabilities of taken actions.
        :return rnn_states: (th.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv).contiguous()

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(
                actor_features, rnn_states)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, available_actions=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (th.Tensor) observation inputs into network.
        :param action: (th.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (th.Tensor) if RNN network, hidden states for RNN.
        :param masks: (th.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (th.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (th.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (th.Tensor) log probabilities of the input actions.
        :return dist_entropy: (th.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions)
        return action_log_probs, dist_entropy
