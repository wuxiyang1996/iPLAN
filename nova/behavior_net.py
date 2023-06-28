import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = f.relu(self.linear(input))
        output, new_hidden = self.rnn(input, hidden)
        new_hidden_out = self.out(new_hidden[-1])
        new_hidden_out = f.softmax(new_hidden_out, dim=-1)
        return output, new_hidden, new_hidden_out


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


class Behavior_Latent_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.5):
        super(Behavior_Latent_Decoder, self).__init__()

        self.decoder = DecoderRNN(input_size, hidden_size, output_size, num_layers, dropout)

    def forward(self, curr_history, prev_latent, hidden):
        # For agent_i, choose history_i [n_thread, max_vehicle_num, max_history_len, obs_dim]
        # Prev latent / Updated latent: [n_thread, max_vehicle_num, latent_dim]

        n_thread, max_vehicle_num, max_history_len, obs_dim = curr_history.shape
        _, _, latent_dim = prev_latent.shape

        latent = prev_latent.reshape(n_thread, max_vehicle_num, 1, latent_dim)
        latent = torch.tile(latent, (1, 1, max_history_len, 1))

        decoder_input = torch.cat([curr_history, latent], dim=-1).reshape(n_thread * max_vehicle_num, max_history_len, -1)

        # Output (Predicted history) [n_thread * max_vehicle_num, max_history_len, obs_dim]
        outputs, hidden = self.decoder(decoder_input, hidden)
        return outputs, hidden
