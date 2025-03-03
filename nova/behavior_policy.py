import torch
import torch.nn as nn
import numpy as np
import os
from nova.behavior_net import EncoderRNN, Behavior_Latent_Decoder
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

    # Initialize the behavior encoder / decoder with their optimizers
    def init_behavior_net(self):
        self.behavior_encoder = []
        self.behavior_decoder = []
        self.behavior_optimizer = []

        for i in range(self.n_agents):
            self.behavior_encoder.append(EncoderRNN(input_size=self.args.obs_shape_single,
                                                    hidden_size=self.args.encoder_rnn_dim,
                                                    output_size=self.args.latent_dim,
                                                    num_layers=self.args.num_encoder_layer).to(self.device))

            self.behavior_decoder.append(Behavior_Latent_Decoder(
                                                    input_size=self.args.obs_shape_single + self.args.latent_dim,
                                                    hidden_size=self.args.decoder_rnn_dim,
                                                    output_size=self.args.obs_shape_single,
                                                    num_layers=self.args.num_decoder_layer,
                                                    dropout=self.args.decoder_dropout).to(self.device))

            self.encoder_parameters = list(self.behavior_encoder[i].parameters())
            self.decoder_parameters = list(self.behavior_decoder[i].parameters())

            self.behavior_optimizer.append(
                torch.optim.Adam(self.encoder_parameters + self.decoder_parameters, lr=self.args.lr_behavior,
                                                    eps=self.optim_eps,
                                                    weight_decay=self.weight_decay))

    # Update the behavior latent in the rollout
    def latent_update(self, history, encoder_hidden, prev_latent):
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
        _, num_layers, _, _, encoder_rnn_dim = encoder_hidden.shape

        for i in range(self.n_agents):
            encoder_hidden_per = torch.Tensor(encoder_hidden[:, :, i]).permute((1, 0, 2, 3)).to(self.device)
            encoder_hidden_per = encoder_hidden_per.reshape(num_layers, n_thread * max_vehicle_num, encoder_rnn_dim)

            history_per = history[:, i, :, :, :].reshape(-1, max_history_len, obs_dim)
            history_per = torch.Tensor(history_per).to(self.device)

            out, encoder_hidden_per, latent = self.behavior_encoder[i](history_per, encoder_hidden_per)
            new_latent.append(latent.reshape(n_thread, max_vehicle_num, -1).unsqueeze(1))

            encoder_hidden_per = encoder_hidden_per.reshape(num_layers, n_thread, max_vehicle_num, encoder_rnn_dim)
            encoder_hidden_new.append(encoder_hidden_per.permute((1, 0, 2, 3)).unsqueeze(2))

        new_latent = torch.cat(new_latent, dim=1)
        new_latent = new_latent.cpu().detach().numpy()

        encoder_hidden = torch.cat(encoder_hidden_new, dim=2)
        encoder_hidden.cpu().detach().numpy()

        return new_latent, encoder_hidden

    # Compare the current history with max_history_len and next history with max_history_len
    # Compute the MSE loss between the two as the input of VAE
    def learn(self, batch, t_env):
        train_info = {
            'behavior_loss': 0.,
            "behavior_encoder_grad_norm": 0.,
            "behavior_decoder_grad_norm": 0.,
        }

        behavior_loss = []

        # history: [n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim]
        # agent_terminate: [n_thread, max_episode_len, n_agent, 1]
        history = batch["history"][:, :-1]
        agent_terminate = batch["terminated"][:, :-1]
        n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim = history.shape

        for i in range(self.n_agents):
            # Agent history: [n_thread, max_episode_len, max_vehicle_num, obs_dim]
            # => [n_thread, num_history, max_history_len, max_vehicle_num, obs_dim]
            agent_history = history[:, :, i, :, :]

            # Determine the update time point, create teh history for hard updating policy
            # Note: the number of folded episode history is num_history - 1, same for the mask size
            num_history = int(max_episode_len // self.max_history_len)
            agent_history = agent_history.reshape(n_thread, num_history, self.max_history_len, max_vehicle_num, obs_dim)

            cut_len = (num_history - 1) * self.max_history_len

            if self.args.env == "MPE":
                mask_cut = 1 - agent_terminate[:, :cut_len, i, 0]
            else:
                mask_cut = agent_terminate[:, :cut_len, i, 0]

            # Mask over given episode
            mask = mask_cut.reshape(n_thread, cut_len, 1, 1).tile(1, 1, max_vehicle_num, obs_dim)
            mask = torch.Tensor(mask).to(self.device)

            # Prev latent / Updated latent: [n_thread, max_vehicle_num, latent_dim]
            latent = torch.zeros((n_thread, max_vehicle_num, self.latent_dim)).to(self.device)

            # Initialize the hidden state
            encoder_hidden = torch.zeros((self.args.num_encoder_layer, n_thread * self.max_vehicle_num,
                                          self.args.encoder_rnn_dim)).to(self.device)
            decoder_hidden = torch.zeros((self.args.num_decoder_layer, n_thread * self.max_vehicle_num,
                                          self.args.decoder_rnn_dim)).to(self.device)

            pred_history_all = []
            for j in range(num_history - 1):
                # curr history: [n_thread, max_vehicle_num, max_history_len, obs_dim]
                curr_history = torch.Tensor(agent_history[:, j, :, :, :]).permute((0, 2, 1, 3)).to(self.device)

                # Decoder
                pred_history, decoder_hidden = self.behavior_decoder[i](curr_history, latent, decoder_hidden)
                pred_history = pred_history.reshape(n_thread, max_vehicle_num, self.max_history_len, obs_dim)
                pred_history_all.append(pred_history.permute((0, 2, 1, 3)).unsqueeze(1))

                # Encoder
                curr_history_en = curr_history.reshape(n_thread * max_vehicle_num, self.max_history_len, obs_dim)
                _, encoder_hidden, latent = self.behavior_encoder[i](curr_history_en, encoder_hidden)
                latent = latent.reshape(n_thread, max_vehicle_num, self.latent_dim)

            # Transform the history for loss computation
            next_history = agent_history[:, 1:, :, :, :].reshape(n_thread, (num_history - 1) * self.max_history_len, max_vehicle_num, obs_dim)
            pred_history = torch.cat(pred_history_all, dim=1).reshape(n_thread, (num_history - 1) * self.max_history_len, max_vehicle_num, obs_dim)

            # Behavior Error: [n_thread, (num_history - 1) * max_history_len, max_vehicle_num, obs_dim]
            error = torch.mul(torch.abs(next_history - pred_history), mask)
            # Average over history number and threads
            loss = error.sum() / (mask.sum() + EPS) * obs_dim * max_vehicle_num

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

            behavior_loss.append(loss.cpu().detach().numpy())

            # Update the training metrics for tensorboard
            train_info['behavior_loss'] += loss.item()
            train_info['behavior_encoder_grad_norm'] += behavior_encoder_grad_norm.item()
            train_info['behavior_decoder_grad_norm'] += behavior_decoder_grad_norm.item()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in train_info.items():
                self.logger.log_stat(self.log_prefix + k, v, t_env)

        return behavior_loss

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
