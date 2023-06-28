import numpy as np
from envs.mpe.core import World, Agent, Landmark
from envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_normal_agents = args.num_normal_agents
        world.num_tiny_agents = args.num_tiny_agents
        world.num_bulky_agents = args.num_bulky_agents
        world.num_random_agents = args.num_random_agents
        world.num_landmarks = args.num_landmarks
        world.collaborative = True

        world.init_sample_size = args.init_sample_size

        world.world_size = args.world_size

        # add agents

        world.normal_agents = [Agent() for i in range(world.num_normal_agents)]
        obj_id = 0
        for i, agent in enumerate(world.normal_agents):
            agent.name = obj_id
            agent.collide = True
            agent.silent = True
            agent.size = 0.08
            agent.step_size = 1.0
            agent.type = "Normal"
            obj_id += 1

        world.tiny_agents = [Agent() for i in range(world.num_tiny_agents)]
        for i, agent in enumerate(world.tiny_agents):
            agent.name = obj_id
            agent.collide = True
            agent.silent = True
            agent.size = 0.10
            agent.step_size = 0.9
            agent.type = "Tiny"
            obj_id += 1

        world.bulky_agents = [Agent() for i in range(world.num_bulky_agents)]
        for i, agent in enumerate(world.bulky_agents):
            agent.name = obj_id
            agent.collide = True
            agent.silent = True
            agent.size = 0.06
            agent.step_size = 1.1
            agent.type = "Bulky"
            obj_id += 1

        world.random_agents = [Agent() for i in range(world.num_random_agents)]
        for i, agent in enumerate(world.random_agents):
            agent.name = obj_id
            agent.collide = True
            agent.silent = True
            agent.size = 0.08
            agent.step_size = 1.0
            agent.type = "Normal"
            obj_id += 1

        world.agents = world.normal_agents + world.tiny_agents + world.bulky_agents + world.random_agents

        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = obj_id
            landmark.collide = False
            landmark.movable = False
            obj_id += 1

        world.num_agents = len(world.agents)
        # goal_assign = np.random.choice(world.num_agents, size=world.num_agents, replace=False)

        # for i, agent in enumerate(world.agents):
        #     agent.goal = world.landmarks[goal_assign[i]].name

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.world_size, +world.world_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.randint(-world.init_sample_size + 1, world.init_sample_size, world.dim_p) \
                / world.init_sample_size * world.world_size
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        # dists = 10000
        # for l in world.landmarks:
        #     if l.name == agent.goal:
        #         dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        #         rew -= dists
        #         if dists < 0.1:
        #             rew += 100
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        rew -= min(dists)

        if dists < 0.1:
            occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 5
                    collisions += 1
        if collisions > 0:
            if min(dists) < 0.1:
                rew += 10

            dist_all = max([min([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]) \
                           for a in world.agents])

            if dist_all < 0.1:
                rew += 100

        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # local reward
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        rew -= min(dists)
        # # global reward
        #
        # for l in world.landmarks:
        #     if l.name == agent.goal:
        #         dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))), for l in world.landmarks]
        #         rew -= dists
        #
        #         if dists < 0.1:
        #             rew += 100

        collison_flg = False
        # collisions penalty
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        rew -= 5
                        collison_flg = True

        if not collison_flg:
            if min(dists) < 0.1:
                rew += 10

            dist_all = max([min([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]) \
                           for a in world.agents])

            if dist_all < 0.1:
                rew += 100

        return rew

    def observation(self, agent, world):
        obj_num = len(world.landmarks) + len(world.agents)
        obs = np.zeros((obj_num, 5))
        obs[0, 0] = agent.name
        obs[0, 1:3] = agent.state.p_pos
        obs[0, 3:5] = agent.state.p_vel
        idx = 1

        for other in world.agents:
            if other is agent:
                continue
            obs[idx, 0] = other.name
            obs[idx, 1:3] = other.state.p_pos - agent.state.p_pos
            obs[idx, 3:5] = other.state.p_vel
            idx += 1

        for entity in world.landmarks:  # world.entities:
            obs[idx, 0] = entity.name
            obs[idx, 1:3] = entity.state.p_pos - agent.state.p_pos
            idx += 1

        return obs
