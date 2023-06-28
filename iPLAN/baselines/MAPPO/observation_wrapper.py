import numpy as np
from collections import deque

# Multi-threading version
# Highway obs, history wrapper
class observersation_state_history_wrapper:
    def __init__(self, args, n_agents, max_vehicle_num, max_episode_len, max_history_len):
        self.args = args
        self.max_vehicle_num = max_vehicle_num
        self.max_episode_len = max_episode_len
        self.max_history_len = max_history_len
        self.obs_shape = args.obs_shape_single
        self.n_agents = n_agents
        self.n_threads = args.batch_size_run
        self.curr_t = 0
        self.history = None
        self.agent_id = None
        self.obs_vehicle_id = None
        self.history_out = None
        self.history_episode_out = None

    # Agent profile initialize
    # agent_id: Controllable agents' ID in each threads
    # obs_vehicle_id: Observed vehicles' ID by each agent in each threads
    # history: Recorded vehicles' state
    def agent_obs_profile_init(self, obs):
        obs = np.array(obs)

        n_threads, n_agents, obs_num, obs_dim = obs.shape

        self.agent_id = [[] for _ in range(n_threads)]
        self.history = [{} for _ in range(n_threads)]
        self.obs_vehicle_id = []

        for k in range(n_threads):
            for i in range(n_agents):
                agent_id = int(obs[k, i, 0, 0])

                if agent_id not in self.agent_id[k]:
                    self.agent_id[k].append(agent_id)

                self.history[k][i] = {}

            self.obs_vehicle_id.append([[] for _ in range(self.n_agents)])

        return self.history

    # Pure observation and state wrapper w/o ID, for baselines like IPPO and QMIX, multi-threading
    # input: State: Array (n_threads, 1, state_dim)
    # Obs: Array (n_threads, n_agents, obs_num, obs_dim)
    # Output: array, list of array
    def pure_obs_state_wrapper(self, state, obs):
        n_threads, n_agents, obs_num, obs_dim = np.array(obs).shape
        _, _, state_dim = state.shape
        n_vehicles = int(state_dim // obs_dim)
        new_state = state.reshape(n_threads, 1, n_vehicles, obs_dim)

        new_state = new_state[:, :, :, 1:].reshape((n_threads, -1))
        new_obs = obs[:, :, :, 1:].reshape((n_threads, n_agents, -1))
        return new_state, new_obs

    # Multi thread version
    # Create a list with fixed length to store history
    # If new vehicle is observed, create a new deque
    # If existing vehicle is observed, append to existing deque
    # If vehicle is not observed this time, append a new zero_list
    # Ego state included in the observation
    def obs_history_create(self, obs):
        obs = np.array(obs)
        n_threads, n_agents, obs_num, obs_dim = obs.shape
        for k in range(n_threads):
            for i in range(n_agents):
                agent_id = int(obs[k, i, 0, 0])
                agent_idx = self.agent_id[k].index(agent_id)

                # Observed vehicle's ID in this time step
                observed_id_list = []
                for j in range(obs_num):
                    if np.any(obs[k, i, j, :]):
                        observed_id = int(obs[k, i, j, 0])
                        observed_id_list.append(observed_id)
                        if observed_id not in self.obs_vehicle_id[k][agent_idx]:
                            self.obs_vehicle_id[k][agent_idx].append(observed_id)
                            observed_idx = self.obs_vehicle_id[k][agent_idx].index(observed_id)
                            self.history[k][agent_idx][observed_idx] = deque(maxlen=self.max_episode_len)
                        else:
                            observed_idx = self.obs_vehicle_id[k][agent_idx].index(observed_id)

                        self.history[k][agent_idx][observed_idx].append(obs[k, i, j, 1:].copy())

                # Non-observed, masked with zeros
                for existing_id in self.obs_vehicle_id[k][agent_idx]:
                    if existing_id not in observed_id_list:
                        existing_idx = self.obs_vehicle_id[k][agent_idx].index(existing_id)
                        self.history[k][agent_idx][existing_idx].append(np.zeros_like(obs[k, i, 0, 1:]))
        return self.agent_id, self.obs_vehicle_id, self.history

    # Multi thread version
    # Output for behavior classification and latent generation
    # Use for action execution in the rollout runner
    def obs_history_output(self):
        self.history_out = np.zeros((self.n_threads, self.n_agents, self.max_vehicle_num,
                                     self.max_history_len, self.obs_shape))

        for k in range(self.n_threads):
            for i in range(self.n_agents):
                agent_id = self.agent_id[k][i]
                agent_idx = i

                observed_id_list = self.obs_vehicle_id[k][agent_idx]
                for observed_id in observed_id_list:
                    observed_idx = observed_id_list.index(observed_id)

                    history = self.history[k][agent_idx][observed_idx]
                    for j in range(min(len(history), self.max_history_len)):
                        self.history_out[k, agent_idx, observed_idx, self.max_history_len - 1 - j, :] \
                            = self.history[k][agent_idx][observed_idx][len(history) - j - 1].copy()

        return self.history_out

    # Multi thread version
    # Output for trajectory prediction when selecting actions
    # Use for action execution in the rollout runner
    def obs_single_history_output(self):
        self.single_history_out = np.zeros((self.n_threads, self.n_agents, self.max_vehicle_num, self.obs_shape))

        for k in range(self.n_threads):
            for i in range(self.n_agents):
                agent_id = self.agent_id[k][i]
                agent_idx = i

                observed_id_list = self.obs_vehicle_id[k][agent_idx]
                for observed_id in observed_id_list:
                    observed_idx = observed_id_list.index(observed_id)

                    history = self.history[k][agent_idx][observed_idx]
                    for j in range(min(len(history), self.max_history_len)):
                        self.single_history_out[k, agent_idx, observed_idx, :] \
                            = self.history[k][agent_idx][observed_idx][-1].copy()

        return self.single_history_out

    # Multi thread version
    # Output for episode history segment and latent VAE training
    def obs_history_episode_output(self, mask):
        # mask: [n_threads, actual episode_len, n_agents]
        mask = np.array(mask)

        num_history = int(self.max_episode_len // self.max_history_len)

        self.raw_history_episode_out = np.zeros((self.n_threads, self.n_agents, self.max_vehicle_num,
                                                 self.max_episode_len, self.obs_shape))

        for k in range(self.n_threads):
            for i in range(self.n_agents):
                agent_id = self.agent_id[k][i]
                agent_idx = i

                observed_id_list = self.obs_vehicle_id[k][agent_idx]
                for observed_id in observed_id_list:
                    observed_idx = observed_id_list.index(observed_id)

                    history = self.history[k][agent_idx][observed_idx]

                    for j in range(len(history)):
                        self.raw_history_episode_out[k, agent_idx, observed_idx, self.max_episode_len - 1 - j, :] \
                            = self.history[k][agent_idx][observed_idx][len(history) - j - 1].copy() \
                              * (mask[k, self.max_episode_len - j - 1, i])

        self.history_episode_out = self.raw_history_episode_out.reshape(
            (self.n_threads, self.n_agents, self.max_vehicle_num, num_history, self.max_history_len, self.obs_shape))

        return self.raw_history_episode_out, self.history_episode_out
