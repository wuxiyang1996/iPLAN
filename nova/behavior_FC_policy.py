import torch
import torch.nn as nn
import numpy as np
import os
from nova.behavior_FC_net import Encoder_3FC, LILI_Latent_Decoder
from utils.mappo_utils.util import get_grad_norm
import time
import pickle
import copy

EPS = 1e-10

class Behavior_policy:
    def __init__(self, args, logger):
        if args.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.args = args
        self.n_actions = args.n_actions

        # Number of all controllable agents
        self.n_agents = args.n_agents

        # Number of all observed vehicles
        self.max_vehicle_num = args.max_vehicle_num

        # History length stored in the observation wrapper
        self.max_history_len = args.max_history_len

        # Dimension of latent representation
        self.latent_dim = args.latent_dim

        self.optim_eps = args.optim_eps
        self.weight_decay = args.weight_decay

        # A single obs of the kinematic
        self.obs_shape = args.obs_shape

        self.init_behavior_net()

        self.logger = logger
        self.log_prefix = args.log_prefix
        self.log_stats_t = -self.args.learner_log_interval - 1

        self._use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm

        # Stabilization parameters
        self.soft_update_coef = args.soft_update_coef
        self.behavior_variation_penalty = args.behavior_variation_penalty
        self.thres_small_variation = args.thres_small_variation

    # Initialize the behavior encoder / decoder with their optimizers
    def init_behavior_net(self):
        self.behavior_encoder = []
        self.behavior_decoder = []
        self.behavior_optimizer = []

        for i in range(self.n_agents):
            self.behavior_encoder.append(Encoder_3FC(input_size=self.args.obs_shape_single * self.args.max_history_len,
                                                    hidden_size=self.args.encoder_rnn_dim,
                                                    output_size=self.args.latent_dim).to(self.device))

            self.behavior_decoder.append(LILI_Latent_Decoder(
                                                    input_size=self.args.obs_shape_single * self.args.max_history_len + self.args.latent_dim,
                                                    hidden_size=self.args.decoder_rnn_dim,
                                                    output_size=self.args.obs_shape_single * self.args.max_history_len).to(self.device))

            self.encoder_parameters = list(self.behavior_encoder[i].parameters())
            self.decoder_parameters = list(self.behavior_decoder[i].parameters())

            self.behavior_optimizer.append(
                torch.optim.Adam(self.encoder_parameters + self.decoder_parameters, lr=self.args.lr_behavior,
                                                    eps=self.optim_eps,
                                                    weight_decay=self.weight_decay))

    # Update the behavior latent in the rollout
    def latent_update(self, history, encoder_hidden=None, prev_latent=None):
        # Similar to choose action
        # (Batch First)
        # history: [n_thread, n_agent, max_vehicle_num, max_history_len, obs_dim]
        # For agent_i, choose history_i [n_thread, max_vehicle_num, max_history_len, obs_dim]
        # RNN (initial): [n_thread, num_layers, n_agent, max_vehicle_num, encoder_rnn_dim]
        # RNN (per agent): [num_layers, n_thread * max_vehicle_num, encoder_rnn_dim]

        # Prev latent / Updated latent: [n_thread, n_agent, max_vehicle_num, latent_dim] (Output)

        # Batch size: n_thread * max_vehicle_num
        # Seq len: max_history_len
        # H_in: obs_dim
        new_latent = []
        encoder_hidden_new = []
        n_thread, n_agent, max_vehicle_num, max_history_len, obs_dim = history.shape

        for i in range(self.n_agents):
            # history_per: [n_thread, max_vehicle_num, max_history_len, obs_dim]
            history_per = history[:, i, :, :, :].reshape(n_thread, max_vehicle_num, max_history_len * obs_dim)
            history_per = torch.Tensor(history_per).to(self.device)
            latent = self.behavior_encoder[i](history_per)
            new_latent.append(latent.unsqueeze(1))

        new_latent = torch.cat(new_latent, dim=1)
        new_latent = new_latent.cpu().detach().numpy()
        return new_latent, encoder_hidden

    # Use this as the wrapper for behavior batch generation
    # Sample over existing batches
    # Load: History, mask (If agent terminates), attention latent
    def behavior_traj_wrapper(self, history, step, mask):
        # Input here is the recorded history and masks for a single agent
        # Sampled to reduce the load
        # history: [n_thread, max_episode_len, max_vehicle_num, obs_dim]
        # behavior_latent: [n_thread, max_episode_len, max_vehicle_num, latent_dim]
        # Attention rnn (From episode batch): [n_thread, max_episode_len, max_vehicle_num, attention_dim]
        # Mask: [n_thread, max_episode_len]
        n_thread, max_episode_len, max_vehicle_num, obs_dim = history.shape

        curr_traj = torch.zeros((n_thread, max_vehicle_num, self.max_history_len, obs_dim))
        next_traj = torch.zeros((n_thread, max_vehicle_num, self.max_history_len, obs_dim))

        mask_over_curr_traj = torch.ones_like(curr_traj)
        mask_over_next_traj = torch.ones_like(next_traj)

        start_idx = max(0, step - self.max_history_len + 1)
        plug_in_idx = max(0, self.max_history_len - step - 1)

        new_start_idx = max(0, step - self.max_history_len + 2)
        new_plug_in_idx = max(self.max_history_len - step - 2, 0)

        curr_traj[:, :, plug_in_idx:, :] = history[:, start_idx:step + 1, :, :].permute((0, 2, 1, 3))
        next_traj[:, :, new_plug_in_idx:, :] = history[:, new_start_idx:step + 2, :, :].permute((0, 2, 1, 3))

        for i in range(n_thread):
            for j in range(start_idx, step + 1):
                mask_over_curr_traj[i, :, plug_in_idx + j - start_idx, :] = torch.ones((max_vehicle_num, obs_dim)).to(self.device) * mask[i, j]

            for j in range(new_start_idx, step + 2):
                mask_over_curr_traj[i, :, new_plug_in_idx + j - new_start_idx, :] = torch.ones((max_vehicle_num, obs_dim)).to(self.device) * mask[i, j]

        return curr_traj, next_traj, mask_over_curr_traj, mask_over_next_traj

    # Compare the current history with max_history_len and next history with max_history_len
    # Compute the MSE loss between the two as the input of VAE
    def learn(self, batch, t_env):
        train_info = {
            'behavior_loss': 0.,
            "stability_loss": 0.,
            "behavior_total": 0.,
            "behavior_encoder_grad_norm": 0.,
            "behavior_decoder_grad_norm": 0.,
        }

        behavior_loss = []
        stability_loss = []
        total_loss = []
        # history: [n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim]
        # behavior_latent: [n_thread, max_episode_len, n_agent, max_vehicle_num, latent_dim]
        # agent_terminate: [n_thread, max_episode_len, n_agent, 1]
        history = batch["history"][:, :-1]
        behavior_latent = batch["behavior_latent"][:, :-1]
        agent_terminate = batch["terminated"][:, :-1]

        n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim = history.shape
        latent_dim = behavior_latent.shape[-1]

        for i in range(self.n_agents):
            # Agent history: [n_thread, max_episode_len, max_vehicle_num, obs_dim]
            # => [n_thread, num_history, max_history_len, max_vehicle_num, obs_dim]
            agent_history = history[:, :, i]
            agent_behavior_latent = behavior_latent[:, :, i]
            mask = agent_terminate[:, :, i, 0]

            # Prev latent / Updated latent: [n_thread, max_vehicle_num, latent_dim]
            latent = torch.zeros((n_thread, max_vehicle_num, self.latent_dim)).to(self.device)

            # Initialize the hidden state
            behavior_error = 0.
            for j in range(max_episode_len - 1 - self.max_history_len):
                # Wrap the current and next trajectories with their mask for training
                curr_traj, next_traj, mask_over_curr_traj, mask_over_next_traj = \
                    self.behavior_traj_wrapper(agent_history, j, mask)

                # curr history: [n_thread, max_vehicle_num, max_history_len, obs_dim]
                curr_history = torch.Tensor(curr_traj).to(self.device)
                next_history = torch.Tensor(next_traj).to(self.device)

                mask_over_next_traj = torch.Tensor(mask_over_next_traj).to(self.device)

                # Decoder
                pred_history = self.behavior_decoder[i](curr_history, latent)
                pred_history = pred_history.reshape(n_thread, max_vehicle_num, self.max_history_len, obs_dim)

                # Encoder
                curr_history_en = curr_history.reshape(n_thread, max_vehicle_num, self.max_history_len * obs_dim)
                latent = self.behavior_encoder[i](curr_history_en)

                # History Error: [n_thread, max_vehicle_num, max_history_len, obs_dim]
                error = torch.mul(torch.abs(next_history - pred_history), mask_over_next_traj)
                # Average over history number and threads
                behavior_error += error.sum() / (mask_over_next_traj.sum() + EPS) * obs_dim * max_vehicle_num

            behavior_error = behavior_error / (max_episode_len - 1 - self.max_history_len)
            loss = behavior_error

            self.behavior_optimizer[i].zero_grad()
            loss.backward()

            # Apply max_grad_norm
            if self._use_max_grad_norm:
                behavior_encoder_grad_norm = nn.utils.clip_grad_norm_(self.behavior_encoder[i].parameters(), self.max_grad_norm)
            else:
                behavior_encoder_grad_norm = get_grad_norm(self.behavior_encoder[i].parameters())

            if self._use_max_grad_norm:
                behavior_decoder_grad_norm = nn.utils.clip_grad_norm_(self.behavior_decoder[i].parameters(), self.max_grad_norm)
            else:
                behavior_decoder_grad_norm = get_grad_norm(self.behavior_decoder[i].parameters())

            self.behavior_optimizer[i].step()

            behavior_loss.append(behavior_error.cpu().detach().numpy())
            total_loss.append(loss.cpu().detach().numpy())

            # Update the training metrics for tensorboard
            train_info['behavior_loss'] += behavior_error.item()
            train_info['behavior_total'] += loss.item()
            train_info['behavior_encoder_grad_norm'] += behavior_encoder_grad_norm.item()
            train_info['behavior_decoder_grad_norm'] += behavior_decoder_grad_norm.item()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in train_info.items():
                self.logger.log_stat(self.log_prefix + k, v, t_env)

        return behavior_loss, stability_loss, total_loss

    # Save models
    def save_models(self, path):
        for i, behavior_encoder in enumerate(self.behavior_encoder):
            torch.save(behavior_encoder.state_dict(), f"{path}/behavior_encoder_{i}.th")
        for i, behavior_decoder in enumerate(self.behavior_decoder):
            torch.save(behavior_decoder.state_dict(), f"{path}/behavior_decoder_{i}.th")
        for i in range(self.n_agents):
            torch.save(self.behavior_optimizer[i].state_dict(), "{}/behavior_optimizer_{}_opt.th".format(path, i))

    # Load models
    def load_models(self, paths:list, load_optimisers=False):
        if len(paths) == 1:
            path = copy.copy(paths[0])
            paths = [path for i in range(self.n_agents)]

        for i, behavior_encoder in enumerate(self.behavior_encoder):
            behavior_encoder.load_state_dict(
                torch.load("{}/behavior_encoder_{}.th".format(paths[i], i),
                            map_location=lambda storage, loc: storage))
        for i, behavior_decoder in enumerate(self.behavior_decoder):
            behavior_decoder.load_state_dict(
                torch.load("{}/behavior_decoder_{}.th".format(paths[i], i),
                        map_location=lambda storage, loc: storage))

        if load_optimisers:
            if len(paths) == 1:
                path = copy.copy(paths[0])
                paths = [path for i in range(self.n_agents)]

            for i in range(self.n_agents):
                self.behavior_optimizer[i].load_state_dict(torch.load("{}/behavior_optimizer_{}_opt.th".format(paths[i], i),
                                                                      map_location=lambda storage, loc: storage))
