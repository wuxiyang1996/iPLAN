import numpy as np


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            assert self.start >= self.finish, "Only exponential decay is currently supported."
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            if self.start < self.finish:
                return min(self.finish, self.start - self.delta * T)
            else:
                return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass


class FlatSchedule():

    def __init__(self,
                 epsilon,
                 noise_coef=None,
                 noise_decay="linear",
                 time_length=None):

        self.epsilon = epsilon
        self.noise_coef = noise_coef 
        self.noise_decay = noise_decay # decay the noise coef to 0 linearly  
        self.time_length = time_length

        if self.noise_coef is not None:
            self.noise_delta = self.noise_coef / self.time_length

    def eval(self, T):
        if self.noise_coef is not None: 
            if self.noise_decay in ["linear"]:
                noise = np.random.rand() * max((self.noise_coef - self.noise_delta * T), 0)
            return self.epsilon - noise

        return self.epsilon
    pass

