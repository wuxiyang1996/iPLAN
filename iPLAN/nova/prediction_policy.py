import torch
import torch.nn as nn
import os
from nova.GAT_Net import GAT_Net
from nova.prediction_net import Prediction_Decoder
from utils.mappo_utils.util import get_grad_norm
import time
import pickle
import copy
import numpy as np

EPS = 1e-10

class Prediction_policy:
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

        # Episode length
        self.max_episode_len = args.episode_limit

        # Batch size for instant incentive inference module training
        self.prediction_batch_size = args.pred_batch_size

        # Prediction length
        self.pred_length = args.pred_length

        self.optim_eps = args.optim_eps
        self.weight_decay = args.weight_decay

        # A single obs of the kinematic
        self.obs_shape = args.obs_shape_single

        self.logger = logger
        self.log_prefix = args.log_prefix
        self.log_stats_t = -self.args.learner_log_interval - 1

        # Whether use the behavioral incentive in instant incentive prediction
        if self.args.GAT_use_behavior:
            self.GAT_input_dim = self.args.obs_shape_single + self.args.latent_dim
        else:
            self.GAT_input_dim = self.args.obs_shape_single

        self.init_GAT_net()

        self._use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm

    # Initialize the instant encoder / decoder with their optimizers
    def init_GAT_net(self):
        self.pred_GAT = []
        self.pred_decoder = []
        self.pred_optimizer = []

        for i in range(self.n_agents):
            self.pred_GAT.append(GAT_Net(input_shape=self.GAT_input_dim, args=self.args).to(self.device))

            #######################################
            self.pred_decoder.append(Prediction_Decoder(
                                    input_size=self.args.obs_shape_single,
                                    hidden_size=self.args.attention_dim,
                                    output_size=self.args.obs_shape_single,
                                    num_layers=1,
                                    pred_length=self.args.pred_length,
                                    teacher_forcing_ratio=self.args.teacher_forcing_ratio,
                                    dropout=self.args.decoder_dropout).to(self.device))

            self.encoder_parameters = list(self.pred_GAT[i].parameters())
            self.decoder_parameters = list(self.pred_decoder[i].parameters())

            self.pred_optimizer.append(
                torch.optim.Adam(self.encoder_parameters + self.decoder_parameters, lr=self.args.lr_predict,
                                                    eps=self.optim_eps,
                                                    weight_decay=self.weight_decay))

    # GAT update for rollout
    # Note: GAT always has only one layer
    def GAT_latent_update(self, history_single, encoder_hidden, behavior_latent=None):
        # history_single: [n_thread, n_agent, max_vehicle_num, obs_dim]
        n_thread, n_agents, max_vehicle_num, obs_shape = history_single.shape
        # behavior_latent: [n_thread, n_agent, max_vehicle_num, latent_dim]
        _, _, _, latent_dim = behavior_latent.shape
        # GAT encoder RNN: [n_thread, n_agent, max_vehicle_num, attention_dim]
        _, _, _, attention_dim = encoder_hidden.shape

        new_hidden = []
        for i in range(n_agents):
            history_single_per = torch.Tensor(history_single[:, i]).to(self.device)
            if self.args.GAT_use_behavior:
                behavior_latent_per = torch.Tensor(behavior_latent[:, i]).to(self.device)
                history_single_per = torch.cat([history_single_per, behavior_latent_per], dim=-1)

            encoder_hidden_per = torch.Tensor(encoder_hidden[:, i]).to(self.device)
            encoder_hidden_per = encoder_hidden_per.reshape(n_thread * max_vehicle_num, attention_dim)

            encoder_hidden_per = self.pred_GAT[i](history_single_per, encoder_hidden_per)
            encoder_hidden_per = encoder_hidden_per.reshape(n_thread, max_vehicle_num, attention_dim)
            encoder_hidden_per = encoder_hidden_per.unsqueeze(1)
            new_hidden.append(encoder_hidden_per)

        # Output: Attention latent: [n_thread, n_agent, max_vehicle_num, attention_dim]
        attention_latent = torch.cat(new_hidden, dim=1)
        attention_latent = attention_latent.cpu().detach().numpy()
        return attention_latent

    # Use this as the wrapper for prediction batch generation
    # Sample over existing batches
    # Load: History, mask (If agent terminates), attention latent
    def prediction_batch_wrapper(self, history, attention_rnn, mask, behavior_latent=None):
        # Input here is the recorded history and masks for a single agent
        # Sampled to reduce the load
        # history: [n_thread, max_episode_len, max_vehicle_num, obs_dim]
        # behavior_latent: [n_thread, n_agent, max_vehicle_num, latent_dim]
        # Attention rnn (From episode batch): [n_thread, max_episode_len, max_vehicle_num, attention_dim]
        # Mask: [n_thread, max_episode_len]
        n_thread, max_episode_len, max_vehicle_num, obs_dim = history.shape
        _, _, _, latent_dim = behavior_latent.shape
        _, _, _, attention_dim = attention_rnn.shape

        input_traj = torch.zeros((self.prediction_batch_size, max_vehicle_num, 1, obs_dim))
        input_attention = torch.zeros((self.prediction_batch_size, max_vehicle_num, 1, attention_dim))
        actual_traj = torch.zeros((self.prediction_batch_size, max_vehicle_num, self.pred_length, obs_dim))

        if self.args.GAT_use_behavior:
            input_latent = torch.zeros((self.prediction_batch_size, max_vehicle_num, 1, latent_dim))
        else:
            input_latent = None

        mask_over_traj = torch.ones_like(actual_traj)

        avail_len = max_episode_len - self.pred_length - 1
        select_idx = np.random.choice(n_thread * avail_len, size=self.prediction_batch_size, replace=False)

        for i in range(self.prediction_batch_size):
            idx = select_idx[i]

            batch_idx = int(idx // avail_len)
            time_idx = int(idx % avail_len)

            input_traj[i, :, :, :] = history[batch_idx, time_idx, :, :].reshape(max_vehicle_num, 1, obs_dim)
            actual_traj[i, :, :, :] = history[batch_idx, time_idx + 1: time_idx + self.pred_length + 1, :, :].permute((1, 0, 2))

            input_attention[i, :, :, :] = attention_rnn[batch_idx, time_idx, :, :].reshape(max_vehicle_num, 1, attention_dim)
            if self.args.GAT_use_behavior:
                input_latent[i, :, :, :] = behavior_latent[batch_idx, time_idx, :, :].reshape(max_vehicle_num, 1, latent_dim)

            for j in range(self.pred_length):
                mask_over_traj[i, :, j, :] = torch.ones((max_vehicle_num, obs_dim)).to(self.device) * mask[batch_idx, time_idx]

        return input_traj, input_attention, input_latent, actual_traj, mask_over_traj

    # Compare the prediction result and the actual trajectory
    # Compute the MSE loss between the two
    def learn(self, batch, t_env):
        train_info = {
            'prediction_loss': 0.,
            "pred_encoder_grad_norm": 0.,
            "pred_decoder_grad_norm": 0.
        }

        prediction_loss = []
        # history: [n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim]
        # behavior_latent: [n_thread, max_episode_len, n_agent, max_vehicle_num, latent_dim]
        # agent_terminate: [n_thread, max_episode_len, n_agent, 1]
        # Attention latent: [n_thread, max_episode_len, n_agent, max_vehicle_num, attention_dim]
        history = batch["history"][:, :-1]
        attention_latent = batch["attention_latent"][:, :-1]
        behavior_latent = batch["behavior_latent"][:, :-1]
        agent_terminate = batch["terminated"][:, :-1]
        n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim = history.shape
        latent_dim = behavior_latent.shape[-1]

        for i in range(n_agent):
            agent_history = history[:, :, i]
            agent_attention_latent = attention_latent[:, :, i]
            agent_behavior_latent = behavior_latent[:, :, i]
            mask = agent_terminate[:, :, i, 0]

            # Input_state: [prediction_batch, max_vehicle_num, 1, obs_dim]
            # Teacher_state and Mask: [prediction_batch, max_vehicle_num, pred_length, obs_dim]
            # Attention: [prediction_batch, max_vehicle_num, 1, obs_dim]
            input_traj, input_attention, input_latent, actual_traj, mask_over_traj = \
                self.prediction_batch_wrapper(agent_history, agent_attention_latent, mask, agent_behavior_latent)

            # Encoder (GAT)
            history_single_per = input_traj.reshape(self.args.pred_batch_size, self.max_vehicle_num, obs_dim)
            encoder_hidden = input_attention.reshape(self.args.pred_batch_size * self.max_vehicle_num,
                                                    self.args.attention_dim)

            history_single_per = torch.Tensor(history_single_per).to(self.device)
            encoder_hidden = torch.Tensor(encoder_hidden).to(self.device)

            if self.args.GAT_use_behavior:
                latent_per = input_latent.reshape(self.args.pred_batch_size, self.max_vehicle_num, latent_dim)
                latent_per = torch.Tensor(latent_per).to(self.device)
                history_single_per = torch.cat([history_single_per, latent_per], dim=-1)

            # Output (Attention after encoding): [prediction_batch * max_vehicle_num, attention_dim]
            encoder_hidden = self.pred_GAT[i](history_single_per, encoder_hidden)

            # Decoder (Seq2Seq)
            # Predicted_state: [batch_Size, max_vehicle_num, pred_length, obs_dim]
            input_traj = torch.Tensor(input_traj).to(self.device)
            actual_traj = torch.Tensor(actual_traj).to(self.device)
            mask_over_traj = torch.Tensor(mask_over_traj).to(self.device)
            pred_traj = self.pred_decoder[i](input_traj, actual_traj, encoder_hidden)

            # Instant Error: [batch_Size, max_vehicle_num, pred_length, obs_dim]
            error = torch.mul(torch.abs(actual_traj - pred_traj), mask_over_traj)
            # Average over batch size and threads
            loss = error.sum() / (mask_over_traj.sum() + EPS) * obs_dim * self.args.pred_length

            self.pred_optimizer[i].zero_grad()
            loss.backward()

            # Apply max_grad_norm
            if self._use_max_grad_norm:
                pred_encoder_grad_norm = nn.utils.clip_grad_norm_(self.pred_GAT[i].parameters(), self.max_grad_norm)
            else:
                pred_encoder_grad_norm = get_grad_norm(self.pred_GAT[i].parameters())

            if self._use_max_grad_norm:
                pred_decoder_grad_norm = nn.utils.clip_grad_norm_(self.pred_decoder[i].parameters(), self.max_grad_norm)
            else:
                pred_decoder_grad_norm = get_grad_norm(self.pred_decoder[i].parameters())

            self.pred_optimizer[i].step()
            prediction_loss.append(loss.cpu().detach().numpy())

            # Update the training metrics for tensorboard
            train_info['prediction_loss'] += loss.item()
            train_info['pred_encoder_grad_norm'] += pred_encoder_grad_norm.item()
            train_info['pred_decoder_grad_norm'] += pred_decoder_grad_norm.item()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in train_info.items():
                self.logger.log_stat(self.log_prefix + k, v, t_env)

        return prediction_loss

    # Save models
    def save_models(self, path):
        for i, pred_GAT in enumerate(self.pred_GAT):
            torch.save(pred_GAT.state_dict(), f"{path}/pred_GAT_{i}.th")
        for i, pred_decoder in enumerate(self.pred_decoder):
            torch.save(pred_decoder.state_dict(), f"{path}/pred_decoder_{i}.th")
        for i in range(self.n_agents):
            torch.save(self.pred_optimizer[i].state_dict(), "{}/pred_optimizer_{}_opt.th".format(path, i))

    # Load models
    def load_models(self, paths:list, load_optimisers=False):
        if len(paths) == 1:
            path = copy.copy(paths[0])
            paths = [path for i in range(self.n_agents)]

        for i, pred_GAT in enumerate(self.pred_GAT):
            pred_GAT.load_state_dict(
                torch.load("{}/pred_GAT_{}.th".format(paths[i], i),
                            map_location=lambda storage, loc: storage))
        for i, pred_decoder in enumerate(self.pred_decoder):
            pred_decoder.load_state_dict(
                torch.load("{}/pred_decoder_{}.th".format(paths[i], i),
                        map_location=lambda storage, loc: storage))

        if load_optimisers:
            if len(paths) == 1:
                path = copy.copy(paths[0])
                paths = [path for i in range(self.n_agents)]

            for i in range(self.n_agents):
                self.pred_optimizer[i].load_state_dict(torch.load("{}/pred_optimizer_{}_opt.th".format(paths[i], i),
                                                                      map_location=lambda storage, loc: storage))