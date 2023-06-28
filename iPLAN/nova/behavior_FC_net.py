import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

class Encoder_3FC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder_3FC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = torch.tanh(self.linear_1(input))
        output = torch.tanh(self.linear_2(input))
        output = f.softmax(self.out(output), dim=-1)
        return output


class Decoder_3FC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder_3FC, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoded_input):
        encoded_input = torch.tanh(self.linear_1(encoded_input))
        decoded_output = torch.tanh(self.linear_2(encoded_input))
        decoded_output = self.out(decoded_output)
        return decoded_output


class LILI_Latent_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LILI_Latent_Decoder, self).__init__()

        self.decoder = Decoder_3FC(input_size, hidden_size, output_size)

    def forward(self, curr_history, prev_latent):
        # For agent_i, choose history_i [n_thread, max_vehicle_num, max_history_len, obs_dim]
        # Prev latent / Updated latent: [n_thread, max_vehicle_num, latent_dim]

        n_thread, max_vehicle_num, max_history_len, obs_dim = curr_history.shape
        _, _, latent_dim = prev_latent.shape

        curr_history = curr_history.reshape(n_thread, max_vehicle_num, -1)
        latent = prev_latent.reshape(n_thread, max_vehicle_num, latent_dim)

        decoder_input = torch.cat([curr_history, latent], dim=-1)

        # Output (Predicted history) [n_thread, max_vehicle_num, max_history_len * obs_dim]
        outputs = self.decoder(decoder_input)
        return outputs
