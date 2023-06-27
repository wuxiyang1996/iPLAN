import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.graphics import EnvViewer
Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        vehicle_num = len(self.road.vehicles)
        for i in range(vehicle_num):
            self.road.vehicles[i].vehicle_id = i

    def get_state(self):
        state = []
        features = self.config["observation"]["observation_config"]['features']
        features_range = self.config["observation"]["observation_config"]['features_range']
        for i in range(len(self.road.vehicles)):
            vec_raw = self.road.vehicles[i].to_dict()
            single_state = []
            for key in features:
                single_state.append(vec_raw[key])
            state.append(single_state.copy())
        return state

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = []
        for i in range(len(self.controlled_vehicles)):
            rewards.append(self.agent_reward(self.controlled_vehicles[i], action))
        self.rewards = rewards.copy()
        return sum(rewards)

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
            info["agents_rewards"] = self.rewards
            info["agents_dones"] = self.dones
            info["agents_terminated"] = self.terminated
        except NotImplementedError:
            pass
        return info

    def agent_reward(self, vehicle, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""

        dones = []
        terminated = []
        for i in range(len(self.controlled_vehicles)):
            terminal = self.controlled_vehicles[i].crashed or \
                       self.time >= self.config["duration"] or \
                       (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road)
            done = self.time >= self.config["duration"] and \
                   (not self.controlled_vehicles[i].crashed) and \
                   (not (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road))

            dones.append(done)
            terminated.append(terminal)
        self.dones = dones.copy()
        self.terminated = terminated.copy()
        return np.all(np.array(dones)), np.all(np.array(terminated))

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        costs = []
        for i in range(len(self.controlled_vehicles)):
            costs.append(float(self.controlled_vehicles[i].crashed))
        self.costs = costs.copy()
        return self.costs


class HighwayEnvHetero(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "normal_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "aggressive_vehicle_type": "highway_env.vehicle.behavior.AggressiveVehicle",
            "defensive_vehicle_type": "highway_env.vehicle.behavior.DefensiveVehicle",
            "ratio_aggressive": 0.1,
            "ratio_defensive": 0.1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["normal_vehicles_type"])
        aggro_type = utils.class_from_path(self.config["aggressive_vehicle_type"])
        defen_type = utils.class_from_path(self.config["defensive_vehicle_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []

        for others in other_per_controlled:
            controlled_vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            controlled_vehicle = self.action_type.vehicle_class(self.road, controlled_vehicle.position,
                                                                controlled_vehicle.heading, controlled_vehicle.speed)
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                random_num = np.random.random()
                if random_num < self.config["ratio_aggressive"]:
                    vehicle = aggro_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                elif random_num > 1 - self.config["ratio_defensive"]:
                    vehicle = defen_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                else:
                    vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        vehicle_num = len(self.road.vehicles)
        for i in range(vehicle_num):
            self.road.vehicles[i].vehicle_id = i

    def get_state(self):
        state = []
        features = self.config["observation"]["observation_config"]['features']
        features_range = self.config["observation"]["observation_config"]['features_range']
        for i in range(len(self.road.vehicles)):
            vec_raw = self.road.vehicles[i].to_dict()
            single_state = []
            for key in features:
                single_state.append(vec_raw[key])
            state.append(single_state.copy())
        return state

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = []
        for i in range(len(self.controlled_vehicles)):
            rewards.append(self.agent_reward(self.controlled_vehicles[i], action))
        self.rewards = rewards.copy()
        return sum(rewards)

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
            info["agents_rewards"] = self.rewards
            info["agents_dones"] = self.dones
            info["agents_terminated"] = self.terminated
        except NotImplementedError:
            pass
        return info

    def agent_reward(self, vehicle, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""

        dones = []
        terminated = []
        for i in range(len(self.controlled_vehicles)):
            terminal = self.controlled_vehicles[i].crashed or \
                       self.time >= self.config["duration"] or \
                       (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road)
            done = self.time >= self.config["duration"] and \
                   (not self.controlled_vehicles[i].crashed) and \
                   (not (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road))

            dones.append(done)
            terminated.append(terminal)
        self.dones = dones.copy()
        self.terminated = terminated.copy()
        return np.all(np.array(dones)), np.all(np.array(terminated))

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        costs = []
        for i in range(len(self.controlled_vehicles)):
            costs.append(float(self.controlled_vehicles[i].crashed))
        self.costs = costs.copy()
        return self.costs

    def render(self, mode: str = 'human'):
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True
        self.viewer.offscreen = False
        self.viewer.observer_vehicle = self.controlled_vehicles[0]
        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            img = []
            for agent in self.controlled_vehicles:
                self.viewer.observer_vehicle = agent
                self.viewer.offscreen = True
                self.viewer.display()
                img.append(self.viewer.get_image())
            image = np.concatenate(img, axis=1)
            return image


class HighwayEnvHetero_H(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "normal_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "aggressive_vehicle_type": "highway_env.vehicle.behavior.AggressiveVehicle",
            "defensive_vehicle_type": "highway_env.vehicle.behavior.DefensiveVehicle",
            "ratio_aggressive": 0.3,
            "ratio_defensive": 0.3,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["normal_vehicles_type"])
        aggro_type = utils.class_from_path(self.config["aggressive_vehicle_type"])
        defen_type = utils.class_from_path(self.config["defensive_vehicle_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []

        for others in other_per_controlled:
            controlled_vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            controlled_vehicle = self.action_type.vehicle_class(self.road, controlled_vehicle.position,
                                                                controlled_vehicle.heading, controlled_vehicle.speed)
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                random_num = np.random.random()
                if random_num < self.config["ratio_aggressive"]:
                    vehicle = aggro_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                elif random_num > 1 - self.config["ratio_defensive"]:
                    vehicle = defen_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                else:
                    vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        vehicle_num = len(self.road.vehicles)
        for i in range(vehicle_num):
            self.road.vehicles[i].vehicle_id = i

    def get_state(self):
        state = []
        features = self.config["observation"]["observation_config"]['features']
        features_range = self.config["observation"]["observation_config"]['features_range']
        for i in range(len(self.road.vehicles)):
            vec_raw = self.road.vehicles[i].to_dict()
            single_state = []
            for key in features:
                single_state.append(vec_raw[key])
            state.append(single_state.copy())
        return state

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = []
        for i in range(len(self.controlled_vehicles)):
            rewards.append(self.agent_reward(self.controlled_vehicles[i], action))
        self.rewards = rewards.copy()
        return sum(rewards)

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
            info["agents_rewards"] = self.rewards
            info["agents_dones"] = self.dones
            info["agents_terminated"] = self.terminated
        except NotImplementedError:
            pass
        return info

    def agent_reward(self, vehicle, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""

        dones = []
        terminated = []
        for i in range(len(self.controlled_vehicles)):
            terminal = self.controlled_vehicles[i].crashed or \
                       self.time >= self.config["duration"] or \
                       (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road)
            done = self.time >= self.config["duration"] and \
                   (not self.controlled_vehicles[i].crashed) and \
                   (not (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road))

            dones.append(done)
            terminated.append(terminal)
        self.dones = dones.copy()
        self.terminated = terminated.copy()
        return np.all(np.array(dones)), np.all(np.array(terminated))

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        costs = []
        for i in range(len(self.controlled_vehicles)):
            costs.append(float(self.controlled_vehicles[i].crashed))
        self.costs = costs.copy()
        return self.costs


    def render(self, mode: str = 'human'):
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True
        self.viewer.offscreen = False
        self.viewer.observer_vehicle = self.controlled_vehicles[0]
        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            img = []
            for agent in self.controlled_vehicles:
                self.viewer.observer_vehicle = agent
                self.viewer.offscreen = True
                self.viewer.display()
                img.append(self.viewer.get_image())
            image = np.concatenate(img, axis=1)
            return image

class HighwayEnvHetero_VH(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "normal_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "aggressive_vehicle_type": "highway_env.vehicle.behavior.AggressiveVehicle",
            "defensive_vehicle_type": "highway_env.vehicle.behavior.DefensiveVehicle",
            "ratio_aggressive": 0.5,
            "ratio_defensive": 0.0,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["normal_vehicles_type"])
        aggro_type = utils.class_from_path(self.config["aggressive_vehicle_type"])
        defen_type = utils.class_from_path(self.config["defensive_vehicle_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []

        for others in other_per_controlled:
            controlled_vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            controlled_vehicle = self.action_type.vehicle_class(self.road, controlled_vehicle.position,
                                                                controlled_vehicle.heading, controlled_vehicle.speed)
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                random_num = np.random.random()
                if random_num < self.config["ratio_aggressive"]:
                    vehicle = aggro_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                elif random_num > 1 - self.config["ratio_defensive"]:
                    vehicle = defen_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                else:
                    vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        vehicle_num = len(self.road.vehicles)
        for i in range(vehicle_num):
            self.road.vehicles[i].vehicle_id = i

    def get_state(self):
        state = []
        features = self.config["observation"]["observation_config"]['features']
        features_range = self.config["observation"]["observation_config"]['features_range']
        for i in range(len(self.road.vehicles)):
            vec_raw = self.road.vehicles[i].to_dict()
            single_state = []
            for key in features:
                single_state.append(vec_raw[key])
            state.append(single_state.copy())
        return state

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = []
        for i in range(len(self.controlled_vehicles)):
            rewards.append(self.agent_reward(self.controlled_vehicles[i], action))
        self.rewards = rewards.copy()
        return sum(rewards)

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
            info["agents_rewards"] = self.rewards
            info["agents_dones"] = self.dones
            info["agents_terminated"] = self.terminated
        except NotImplementedError:
            pass
        return info

    def agent_reward(self, vehicle, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""

        dones = []
        terminated = []
        for i in range(len(self.controlled_vehicles)):
            terminal = self.controlled_vehicles[i].crashed or \
                       self.time >= self.config["duration"] or \
                       (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road)
            done = self.time >= self.config["duration"] and \
                   (not self.controlled_vehicles[i].crashed) and \
                   (not (self.config["offroad_terminal"] and not self.controlled_vehicles[i].on_road))

            dones.append(done)
            terminated.append(terminal)
        self.dones = dones.copy()
        self.terminated = terminated.copy()
        return np.all(np.array(dones)), np.all(np.array(terminated))

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        costs = []
        for i in range(len(self.controlled_vehicles)):
            costs.append(float(self.controlled_vehicles[i].crashed))
        self.costs = costs.copy()
        return self.costs


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class MOHighwayEnv(HighwayEnv):
    """
    A multi-objective version of HighwayEnv
    """

    def _rewards(self, action: Action) -> dict:
        """
        In MORL, we consider multiple rewards like collision, right-keeping, and speed,
        and the utility of these separate rewards is not always known a priori.
        This function returns a dict of multiple reward criteria

        :param action: the last action performed
        :return: the reward vector
        """
        rewards = {}

        rewards["collision"] = self.vehicle.crashed

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        rewards["right_lane"] = lane / max(len(neighbours) - 1, 1)

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        rewards["high_speed"] = np.clip(scaled_speed, 0, 1)

        return rewards

    def _reward(self, action: Action) -> float:
        """
        This scalarized reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.

        :param action: the last action performed
        :return: the reward
        """
        rewards = self._rewards(action)
        reward = \
            + self.config["collision_reward"] * rewards["collision"] \
            + self.config["right_lane_reward"] * rewards["right_lane"] \
            + self.config["high_speed_reward"] * rewards["high_speed"]
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of rewards

        :param obs: current observation
        :param action: current action
        :return: reward dict
        """
        return self._rewards(action)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-hetero-v0',
    entry_point='highway_env.envs:HighwayEnvHetero',
)

register(
    id='highway-hetero-H-v0',
    entry_point='highway_env.envs:HighwayEnvHetero_H',
)

register(
    id='highway-hetero-VH-v0',
    entry_point='highway_env.envs:HighwayEnvHetero_VH',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id='mo-highway-v0',
    entry_point='highway_env.envs:MOHighwayEnv',
)