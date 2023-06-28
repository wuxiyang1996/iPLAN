import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

# Input all obs, get all prob dist
class GAT_Net(nn.Module):
    def __init__(self, input_shape, args):
        super(GAT_Net, self).__init__()
        # Decoding input own h_i and x_i, output the prob dist
        self.args = args
        self.input_shape = input_shape
        self.max_vehicle_num = args.max_vehicle_num
        self.rnn_hidden_dim = args.GAT_hidden_dim
        self.attention_dim = args.attention_dim

        # Encode each observation
        self.encoding = nn.Linear(input_shape, self.rnn_hidden_dim)

        # Hard
        # GRU input: [[h_i,h_1],[h_i,h_2],...[h_i,h_n]] and [0,...,0]
        #     Output[[h_1],[h_2],...,[h_n]] and [h_n],
        # h_j represents the relation between agent j and agent i
        # Input dim = (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)，
        # i.e. For batch_size data, feed each agent's connection with other n_agents - 1 agents' hidden_state
        self.hard_bi_GRU = nn.GRU(self.rnn_hidden_dim * 2, self.rnn_hidden_dim, bidirectional=True)
        # Analyze h_j,get agent j's weight wrt agent i, output dim = 2, grab one of them after gumble_softmax
        # If 1, then consider agent_j
        # Bi-direction GRU, hidden_state dim = 2 * hidden_dim
        self.hard_encoding = nn.Linear(self.rnn_hidden_dim * 2, 2)

        # Soft
        self.q = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.k = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.rnn_hidden_dim, self.attention_dim)

        # Get hidden state from self obs
        # Each agent gets hidden_state from own obs to store prev obs
        self.rnn = nn.GRUCell(self.attention_dim, self.attention_dim)

    def forward(self, obs, hidden_state):
        # For agent_i, choose history_i [batch_size, max_vehicle_num, obs_dim]
        # batch_size = (1 (select action) / max_episode_len) * n_thread
        # batch_size * n_agents
        n_thread, max_vehicle_num, obs_dim = obs.shape
        size = n_thread * max_vehicle_num

        # Encode the obs
        obs_encoding = f.relu(self.encoding(obs))
        # Get h from own GRU, dim = (batch_size * max_vehicle_num, args.rnn_hidden_dim)
        h_out = obs_encoding.reshape(-1, self.rnn_hidden_dim)

        # Hard Attention, GRU input dim = (seq_len, batch_size, dim)
        # Reshape h to have n_agents dim, (batch_size, max_vehicle_num, rnn_hidden_dim)
        h = h_out.reshape(-1, self.max_vehicle_num, self.rnn_hidden_dim)

        input_hard = []
        for i in range(self.max_vehicle_num):
            # (batch_size, rnn_hidden_dim)
            h_i = h[:, i]
            h_hard_i = []
            # For agent i, concatenate h_i wth other agent's h
            for j in range(self.max_vehicle_num):
                if j != i:
                    h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
            # After the loop, h_hard_i is a list with n_agents - 1 tensor
            # with dim = (batch_size, rnn_hidden_dim * 2)
            h_hard_i = torch.stack(h_hard_i, dim=0)
            input_hard.append(h_hard_i)

        # After loop, input_hard is a list with n_agents tensor
        # with dim=(max_vehicle_num - 1, batch_size, max_vehicle_num, rnn_hidden_dim * 2)
        input_hard = torch.stack(input_hard, dim=-2)
        # Finally get (max_vehicle_num - 1, batch_size * max_vehicle_num, rnn_hidden_dim * 2) for input
        input_hard = input_hard.view(self.max_vehicle_num - 1, -1, self.rnn_hidden_dim * 2)

        # Bi-direction GRU, each GRU with 1 layer, so 1st layer is 2 * 1
        h_hard = torch.zeros((2 * 1, size, self.rnn_hidden_dim))
        if self.args.use_cuda:
            h_hard = h_hard.cuda()

        # (max_vehicle_num - 1,batch_size * max_vehicle_num,rnn_hidden_dim * 2)
        h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)
        # (batch_size * max_vehicle_num, max_vehicle_num - 1, rnn_hidden_dim * 2)
        h_hard = h_hard.permute(1, 0, 2)
        # (batch_size * max_vehicle_num * (max_vehicle_num - 1), rnn_hidden_dim * 2)
        h_hard = h_hard.reshape(-1, self.rnn_hidden_dim * 2)

        # Get hard weight, (max_vehicle_num, batch_size, 1,  max_vehicle_num - 1) with an extra dim for sum
        # (batch_size * max_vehicle_num * (max_vehicle_num - 1), 2)
        hard_weights = self.hard_encoding(h_hard)
        # Determine send to agent j or not (one-hot vector)
        hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)

        hard_weights = hard_weights[:, 1].view(-1, self.max_vehicle_num, 1, self.max_vehicle_num - 1)
        # (max_vehicle_num, batch_size, 1, (max_vehicle_num - 1))
        hard_weights = hard_weights.permute(1, 0, 2, 3)

        # Soft Attention
        # (batch_size, max_vehicle_num, args.attention_dim)
        q = self.q(h_out).reshape(-1, self.args.max_vehicle_num, self.attention_dim)
        # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1, self.args.max_vehicle_num, self.attention_dim)
        # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.max_vehicle_num, self.attention_dim)

        x = []
        for i in range(self.max_vehicle_num):
            # agent i's q, (batch_size, 1, args.attention_dim)
            q_i = q[:, i].view(-1, 1, self.attention_dim)
            # Other agent's k and v
            k_i = [k[:, j] for j in range(self.max_vehicle_num) if j != i]
            v_i = [v[:, j] for j in range(self.max_vehicle_num) if j != i]

            # (max_vehicle_num - 1, batch_size, args.attention_dim)
            k_i = torch.stack(k_i, dim=0)
            # Exchange dimensions into (batch_size, args.attention_dim, max_vehicle_num - 1)
            k_i = k_i.permute(1, 2, 0)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim, max_vehicle_num - 1) = (batch_size, 1, max_vehicle_num - 1)
            score = torch.matmul(q_i, k_i)

            # Normalize
            scaled_score = score / np.sqrt(self.attention_dim)

            # softmax to get the weight, dim = (batch_size, 1, max_vehicle_num - 1)
            soft_weight = f.softmax(scaled_score, dim=-1)

            # Weighted sum get (batch_size, args.attention_dim)
            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        # Concatenate each agent's h and x
        # dim = (batch_size * max_vehicle_num, args.attention_dim)
        x = torch.stack(x, dim=1).reshape(-1, self.attention_dim)

        # # Read the hidden state and the attention state to generate the action
        h_out = self.rnn(x, hidden_state)
        # hidden dim = (batch_size * max_vehicle_num, args.attention_dim)
        return h_out

