import torch as th
import torch.nn as nn

from utils.mappo_utils.mlp import MLPBase
from utils.mappo_utils.rnn import RNNLayer
from utils.mappo_utils.popart import PopArt
from utils.mappo_utils.util import init, check


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param device: (th.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, input_shape, args):
        super(R_Critic, self).__init__()
        if args.use_cuda:
            self.device = th.device("cuda")
        else:
            self.device = th.device("cpu")

        self.rnn_hidden_dim = args.rnn_hidden_dim
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=th.float32, device=self.device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.base = MLPBase(args, input_shape)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.rnn_hidden_dim, self.rnn_hidden_dim, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.rnn_hidden_dim, 1, device=self.device))
        else:
            self.v_out = init_(nn.Linear(self.rnn_hidden_dim, 1))

        self.to(self.device)

    def forward(self, obs, rnn_states):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / th.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / th.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / th.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (th.Tensor) value function predictions.
        :return rnn_states: (th.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv).contiguous()

        critic_features = self.base(obs)
        if self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states)
        values = self.v_out(critic_features)

        return values, rnn_states