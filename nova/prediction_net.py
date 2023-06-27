import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        encoded_input = f.relu(self.linear(encoded_input))
        decoded_output, hidden = self.rnn(encoded_input, hidden)
        decoded_output = self.tanh(decoded_output)
        decoded_output = self.dropout(decoded_output)
        decoded_output = self.out(decoded_output)
        return decoded_output, hidden


class Prediction_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, pred_length, dropout=0.5, teacher_forcing_ratio=0.5):
        super(Prediction_Decoder, self).__init__()

        self.pred_length = pred_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size
        self.decoder = DecoderRNN(input_size, hidden_size, output_size, num_layers, dropout)

    # Predict the trajectory for agent team by iteratively feed the agents' state into RNN
    def forward(self, last_state, teacher_state, hidden, prev_latent=None):
        # For agent_i, choose last_state_i [batch_size, max_vehicle_num, 1, obs_dim]
        # Teacher_state: [batch_size, max_vehicle_num, pred_length, obs_dim]
        # Attention (Hidden): [batch_size * max_vehicle_num, attention_dim]
        batch_size, max_vehicle_num, _, obs_dim = last_state.shape
        predicted = torch.zeros(batch_size, max_vehicle_num, self.pred_length, obs_dim).to(last_state.device)
        decoder_input = last_state

        hidden = hidden.reshape(1, batch_size * max_vehicle_num, self.hidden_size)

        for t in range(self.pred_length):
            decoder_input = decoder_input.reshape(batch_size * max_vehicle_num, 1, obs_dim)
            now_out, hidden = self.decoder(decoder_input, hidden)

            now_out = now_out.reshape(batch_size, max_vehicle_num, 1, obs_dim)
            predicted[:, :, t:t + 1, :] = now_out
            teacher_force = False if teacher_state is None else np.random.random() < self.teacher_forcing_ratio

            if teacher_force:
                decoder_input = teacher_state[:, :, t:t + 1, :]
            else:
                decoder_input = now_out

        return predicted
