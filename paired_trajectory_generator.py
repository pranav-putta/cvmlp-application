import math
import pickle
import random
from dataclasses import field

import cv2
import habitat_sim
import numpy as np
import torch
from habitat_sim import ShortestPath
from matplotlib import pyplot as plt
from tqdm import tqdm

from argparser import argclass, parse_args


@argclass
class AgentArguments:
    """
    Arguments for bigfoot/littlefoot agents
    """
    stride_length: float = field(default=1.)
    turn_inc: float = field(default=10.)  # degrees
    goal_threshold: float = field(default=1.)  # max distance from goal for geodesic follower

    @property
    def turn_inc_radians(self):
        return self.turn_inc * torch.pi / 180


@argclass
class Arguments:
    config_file: str = field(default='./trajectory_config.yaml')
    scene: str = field(default='./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb')
    num_samples: int = field(default=10000)  # number of samples to generate
    num_points_per_curve: int = field(default=10)  # number of points in interpolation for curved path
    ratio_curved: float = field(default=0)  # what ratio of trajectories should be curved
    agent_end_max_dist: float = field(default=0.5)  # maximum distance between agents at the end of the path
    min_path_dist: float = field(default=5.)  # the minimum distance of the path for each trajectory
    min_curvature_ratio: float = field(default=1.2)  # defined by straight-line to geodesic distance
    save_loc: str = field(default='./trajectory_straight.pkl')  # data save location
    bigfoot: AgentArguments
    littlefoot: AgentArguments


def show_observation(observations, key='color_sensor'):
    """
    draw observation to screen
    :param observations:
    :param key:
    :return:
    """
    image = observations[key]
    image = image[:, :, [2, 1, 0]]
    cv2.imshow('RGB', image)


def make_cfg(args: Arguments):
    """
    creates simulator and agents
    :param args: arguments
    :return:
    """
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = args.scene

    # agent
    bigfoot_agent = habitat_sim.agent.AgentConfiguration()
    littlefoot_agent = habitat_sim.agent.AgentConfiguration()

    bigfoot_agent.radius = 0.5
    bigfoot_agent.action_space['move_forward'].actuation.amount = args.bigfoot.stride_length
    bigfoot_agent.action_space['turn_left'].actuation.amount = args.bigfoot.turn_inc
    bigfoot_agent.action_space['turn_right'].actuation.amount = args.bigfoot.turn_inc

    littlefoot_agent.radius = 0.5
    littlefoot_agent.action_space['move_forward'].actuation.amount = args.littlefoot.stride_length
    littlefoot_agent.action_space['turn_left'].actuation.amount = args.littlefoot.turn_inc
    littlefoot_agent.action_space['turn_right'].actuation.amount = args.littlefoot.turn_inc

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    def create_camera_spec():
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f"color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [256, 256]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        rgb_sensor_spec.orientation = [0, 0, 0]
        return rgb_sensor_spec

    bigfoot_agent.sensor_specifications = [create_camera_spec()]
    littlefoot_agent.sensor_specifications = [create_camera_spec()]

    return habitat_sim.Configuration(sim_cfg, [bigfoot_agent, littlefoot_agent])


def quaternion_from_euler(r):
    """
    compute quaternion from an euler angle vector
    :param r: angle vec, axes: [xy, xz, yz]
    :return: np.quaternion
    """
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return np.quaternion(qw, qx, qy, qz)


def init_agent(sim: habitat_sim.Simulator, agent_id):
    """
    sets up agent -- initial point doesn't really matter since agent
    will be teleported to start point when starting on a path
    :param sim: habitat simulator
    :param agent_id: which agent to initialize
    :return:
    """
    agent = sim.initialize_agent(agent_id)

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.9539339, 0.1917877, 10.163067])
    agent_state.rotation = quaternion_from_euler([0, -torch.pi, 0])
    agent.set_state(agent_state)


def get_shortest_path(sim: habitat_sim.Simulator, start, end):
    """
    computes shortest geodesic path from navmesh
    :param sim: habitat simulator
    :param start: start point
    :param end: end point
    :return:
    """
    shortest_path = ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = end
    sim.pathfinder.find_path(shortest_path)
    return shortest_path.points


def unit(vec):
    """
    computes unit vector
    :param vec: vector
    :return:
    """
    norm = np.sqrt(vec @ vec)
    if norm == 0:
        return vec * 0
    return vec / np.sqrt(vec @ vec)


def angle_between_vecs(a, b):
    """
    2D clockwise angle between vectors
    :param a: vec 1
    :param b: vec 2
    :return:
    """
    a = unit(a)
    b = unit(b)

    dot = np.dot(a, b)
    det = a[0] * b[1] - b[0] * a[1]
    return np.arctan2(det, dot)


def add_curvature_to_path(args: Arguments, sim: habitat_sim.Simulator, path):
    """
    creates a curved path by adding intermediate points to path
    :param args: arguments
    :param sim: habitat simulator
    :param path: current path of length 2, should be [start, goal]
    :return:
    """
    start, end = path
    start, end = np.array(start), np.array(end)
    start_xz, end_xz = start[[0, 2]], end[[0, 2]]
    diff_xz = end_xz - start_xz

    ctrl_point = lambda _x: (_x, 1 - _x)  # control point to determine curve
    quad = lambda _a, _b, _x: _a * _x ** 2 + _b * _x
    d_quad = lambda _a, _b, _x: 2 * _a * _x + _b

    n = 100
    approx_arc_length = lambda _a, _b: sum([(1 / n) * math.sqrt(1 + d_quad(_a, _b, _x) ** 2) for _x in
                                            np.arange(0, 1 + 1 / n, 1 / n)])  # approximate arc length from [0, 1]
    points = []
    for x in np.arange(0.8, 0.5, -0.05):
        px, py = ctrl_point(x)

        a = (py - px) / (px ** 2 - px)
        b = 1 - a
        if approx_arc_length(a, b) < args.min_curvature_ratio:
            # if arc length is too small, reject this (start, goal) pair and move on
            return None

        # compute points
        new_points = [np.array([_x, quad(a, b, _x)]) for _x in np.arange(0, 1 + 1 / args.num_points_per_curve,
                                                                         1 / args.num_points_per_curve)]

        # transform to original space
        theta = angle_between_vecs(np.array([1, 1]), diff_xz)
        # quadratic moves from (0,0) -> (1,1) this curve is less boring than a symmetric one / adds more variation
        #                                                                                       to the dataset
        scale_matrix = np.eye(2) * np.sqrt(diff_xz @ diff_xz) / math.sqrt(2)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # scale then rotate (order doesn't matter, scale+rot commutative)
        transf_matrix = rot_matrix @ scale_matrix

        points = [start_xz + transf_matrix @ point for point in new_points]

        # ensure all points are navigable
        points = [np.array([point[0], start[1], point[1]]) for point in points]
        navigable = np.array([sim.pathfinder.is_navigable(point) for point in points]).all()
        if not navigable:
            # rip make curve smaller
            points = None
            continue
    return points


def generate_path(args: Arguments, sim: habitat_sim.Simulator, curved=False):
    """
    generate a random navigable path in the environment.
    :param args: arguments
    :param sim: habitat simulator
    :param curved: if True, adds curvature to path
    :return:
    """
    path = []
    while not path:
        start = sim.pathfinder.get_random_navigable_point()
        end = sim.pathfinder.get_random_navigable_point()

        # make sure Y-axis is set to constant since agent can't manipulate this
        start[1] = 0.15
        end[1] = start[1]

        if dist(start, end) < args.min_path_dist:
            continue

        if get_shortest_path(sim, start, end):
            path = [start, end]
            if curved:
                path = add_curvature_to_path(args, sim, path)
    return path


def dist(u, v):
    """
    computes distance between two vectors, this might be a little crude,
     but forgot to convert some things to numpy arrays and don't want to debug rn whoops
    :param u: vec 1
    :param v: vec 2
    :return:
    """
    dx = u[0] - v[0]
    dy = u[1] - v[1]
    dz = u[2] - v[2]

    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def follow_path(sim: habitat_sim.Simulator, agent_id, path, goal_threshold=0.1, step_by_key=False):
    """
    makes agent follow desired path by using the GreedyGeodesicFollower
    :param sim: habitat simulator
    :param agent_id: agent to move
    :param path: path to follow
    :param goal_threshold: max distance from each goal point
    :param step_by_key: for debugging, allows stepping through frames by pressing a key
    :return:
    """
    # set agent to start node
    agent = sim.get_agent(agent_id)

    # ensure distance between points in path are greater than agent's stride length, filter out extra intermediate steps
    tmp = [path[0]]
    for point in path[1:-1]:
        if dist(tmp[-1], point) >= agent.agent_config.action_space['move_forward'].actuation.amount:
            tmp.append(point)
    tmp.append(path[-1])
    path = tmp

    realxs = [p[0] for p in path]
    realys = [p[2] for p in path]

    state = agent.get_state()
    state.position = path[0]
    agent.set_state(state)

    follower = sim.make_greedy_follower(agent_id, goal_threshold, forward_key='move_forward',
                                        left_key='turn_left', right_key='turn_right')
    observations = []
    for point in path[1:]:
        try:
            action = follower.next_action_along(point)
        except:
            plt.scatter(realxs, realys, c='r')
            plt.show()
            print(f"greedy follower error, agent: {agent_id}")
            return None
        i = 0
        while action is not None:
            if i >= 50:
                # something has gone wrong with greedy follower ; maybe thrashing or missing goals?
                plt.scatter(realxs, realys, c='r')
                plt.show()
                print(f"greedy follower took too long, agent: {agent_id}")
                return None
            agent.act(action)
            obs = sim.get_sensor_observations(agent_id)
            observations.append((agent.get_state().position, agent.get_state().rotation))
            if step_by_key:
                show_observation(obs)
                cv2.waitKey(0)
            try:
                action = follower.next_action_along(point)
            except:
                plt.scatter(realxs, realys, c='r')
                plt.show()
                print(f"greedy follower error, agent: {agent_id}")
                return None
            i += 1

    if step_by_key:
        xs = [o[0][0] for o in observations]
        ys = [o[0][2] for o in observations]

        plt.plot(xs, ys)
        plt.scatter(realxs, realys, c='r')
        plt.show()
    return observations


def generate_paired_trajectory(args: Arguments, sim: habitat_sim.Simulator, bigfoot_id, littlefoot_id,
                               curved=False, step_by_key=False):
    """
    creates a single data point, paired trajectory
    :param args: arguments
    :param sim: habitat simulator
    :param bigfoot_id: simulator agent_id for bigfoot
    :param littlefoot_id: simulator agent_id for littlefoot
    :param curved: if True, generates a curved path
    :param step_by_key: for debugging, allows for stepping through frames by hitting a key
    :return: (bigfoot_trajectory, littlefoot_trajectory) if a viable point with good
            end distance is found, otherwise returns None
    """
    path = generate_path(args, sim, curved)
    bigfoot_obs = follow_path(sim, bigfoot_id, path, args.bigfoot.goal_threshold, step_by_key=step_by_key)
    littlefoot_obs = follow_path(sim, littlefoot_id, path, args.littlefoot.goal_threshold, step_by_key=step_by_key)
    if bigfoot_obs and littlefoot_obs:
        bpos = sim.get_agent(bigfoot_id).get_state().position
        lpos = sim.get_agent(littlefoot_id).get_state().position
        print(f'dist btw agent ends: {dist(bpos, lpos)}')
        if dist(bpos, lpos) <= args.agent_end_max_dist:
            return bigfoot_obs, littlefoot_obs
    return None


def manual(sim: habitat_sim.Simulator):
    """
    for debugging purposes, manually manipulate agent
    :param sim: habitat simulator
    :return:
    """
    actions = ['move_forward', 'turn_left', 'turn_right']

    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"

    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = 0
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = 1
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = 2
        print("action: RIGHT")
    else:
        print("INVALID KEY")
        return None
    return sim.step(actions[action])


def main():
    args: Arguments = parse_args(Arguments, resolve_config=False)

    args.bigfoot.stride_length = 3
    args.bigfoot.turn_inc = 30
    args.bigfoot.goal_threshold = args.bigfoot.stride_length / 2 + 0.05

    args.littlefoot.stride_length = 1.25
    args.littlefoot.turn_inc = 20
    args.littlefoot.goal_threshold = args.littlefoot.stride_length / 2 + 0.05

    cfg = make_cfg(args)
    sim = habitat_sim.Simulator(cfg)
    sim.reset()

    init_agent(sim, 0)
    init_agent(sim, 1)
    observations = sim.get_sensor_observations([0, 1])

    show_observation(observations[0], 'color_sensor')

    data = []
    with tqdm(total=args.num_samples) as pbar:
        i = 0
        while i < args.num_samples:
            curved = random.uniform(0, 1) <= args.ratio_curved
            out = generate_paired_trajectory(args, sim, 0, 1, step_by_key=False, curved=curved)
            if out is not None:
                data.append(out)
                pbar.update(1)
                i += 1

    with open(args.save_loc, 'wb+') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
