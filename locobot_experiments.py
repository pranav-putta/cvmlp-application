import os
import random
import sys
from dataclasses import field

import git
import habitat_sim
import numpy as np
from habitat_sim import ShortestPath
from habitat_sim._ext.habitat_sim_bindings import ManagedRigidObject
from habitat_sim.utils import viz_utils as vut
from scipy.spatial.transform import Rotation

from mltoolkit.argparser import argclass, parse_args

random.seed(0)


@argclass
class Arguments:
    make_video: bool = field(default=True)
    show_video: bool = field(default=True)

    data_path: str = field(default=None)
    output_path: str = field(default=None)

    def __post_init__(self):
        repo = git.Repo(".", search_parent_directories=True)
        dir_path = repo.working_tree_dir
        # %cd $dir_path
        if self.data_path is None:
            self.data_path = os.path.join(dir_path, "data")
        if self.output_path is None:
            self.output_path = os.path.join(
                dir_path, "outputs/tutorials/managed_rigid_object_tutorial_output/"
            )


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    agent_state.rotation = np.quaternion(1, 0, 0, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration(args: Arguments):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(
        args.data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb"
    )
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [544, 720]
    sensor_specs = []

    rgba_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_1stperson_spec.uuid = "rgba_camera_1stperson"
    rgba_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_1stperson_spec.resolution = camera_resolution
    rgba_camera_1stperson_spec.position = [0.0, 0.6, 0.0]
    rgba_camera_1stperson_spec.orientation = [0.0, 0.0, 0.0]
    rgba_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_1stperson_spec)

    depth_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    depth_camera_1stperson_spec.uuid = "depth_camera_1stperson"
    depth_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_camera_1stperson_spec.resolution = camera_resolution
    depth_camera_1stperson_spec.position = [0.0, 0.6, 0.0]
    depth_camera_1stperson_spec.orientation = [0.0, 0.0, 0.0]
    depth_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_camera_1stperson_spec)

    rgba_camera_3rdperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_3rdperson_spec.uuid = "rgba_camera_3rdperson"
    rgba_camera_3rdperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_3rdperson_spec.resolution = camera_resolution
    rgba_camera_3rdperson_spec.position = [0.0, 1.0, 0.3]
    rgba_camera_3rdperson_spec.orientation = [-45, 0.0, 0.0]
    rgba_camera_3rdperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_3rdperson_spec)

    birdseye = habitat_sim.CameraSensorSpec()
    birdseye.uuid = "birdseye"
    birdseye.sensor_type = habitat_sim.SensorType.COLOR
    birdseye.resolution = camera_resolution
    birdseye.position = [2.5, 3.5, -0.5]
    birdseye.orientation = [-89.50, 0.0, 0.0]
    birdseye.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(birdseye)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations


def load_locobot(sim, obj_templates_mgr, rigid_obj_mgr, args: Arguments, agent):
    locobot_template_id = obj_templates_mgr.load_configs(
        str(os.path.join(args.data_path, "objects/locobot_merged"))
    )[0]

    # add robot object to the scene with the agent/camera SceneNode attached
    if agent is not None:
        locobot = rigid_obj_mgr.add_object_by_template_id(
            locobot_template_id, agent.scene_node
        )
    else:
        locobot = rigid_obj_mgr.add_object_by_template_id(
            locobot_template_id, None
        )
    return locobot


def random_point(x_lim=(-10, 10), y_lim=(-10, 10), z_lim=(-10, 10)):
    return list(map(lambda lim: random.uniform(*lim), [x_lim, y_lim, z_lim]))


def get_shortest_path(sim: habitat_sim.Simulator, start, end):
    shortest_path = ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = end
    sim.pathfinder.find_path(shortest_path)
    return shortest_path.points


def angle_between_vecs(a, b):
    # 2d clockwise angle between vecs
    a = unit(a)
    b = unit(b)

    dot = np.dot(a, b)
    det = a[0] * b[1] - b[0] * a[1]
    return np.arctan2(det, dot)


def quaternion_from_euler(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return np.quaternion(qw, qx, qy, qz)


def unit(vec):
    norm = np.sqrt(vec @ vec)
    if norm == 0:
        return vec * 0
    return vec / np.sqrt(vec @ vec)


def rotate_to_face_point(sim, locobot: ManagedRigidObject, goal, args: Arguments, dt=1, step=None):
    start = locobot.translation
    v, w = locobot.rotation.vector, locobot.rotation.scalar
    q = np.quaternion(w, v.x, v.y, v.z)

    z_unit = np.quaternion(0, 0, 0, -1)
    current_vec = q * z_unit * q.conjugate()
    current_vec = unit(np.array([current_vec.x, current_vec.y, current_vec.z]))
    dir_vec = unit(np.array(goal) - start)

    # compute angles from unit vectors for goal : we want to face in the direction of origin -> goal
    d_xy = angle_between_vecs(current_vec[[0, 1]], dir_vec[[0, 1]])
    d_yz = angle_between_vecs(current_vec[[1, 2]], dir_vec[[1, 2]])
    d_xz = angle_between_vecs(current_vec[[0, 2]], dir_vec[[0, 2]])
    diff_euler = np.array([0 * d_xy, -d_xz, 0 * d_yz])

    a = 4 * diff_euler / (dt ** 2)

    def angular_vel_interp(t):
        if t < dt / 2:
            return a * t
        else:
            return -a * t + a * dt

    start_time = sim.get_world_time()

    v, w = locobot.rotation.vector, locobot.rotation.scalar
    current_euler = Rotation.from_quat([v.x, v.y, v.z, w]).as_euler('xyz')
    observations = []
    if step is None:
        step = 0
        while sim.get_world_time() < start_time + dt:
            locobot.velocity_control.angular_velocity = angular_vel_interp(step)
            sim.step_physics(1.0 / 60.0)
            if args.make_video:
                observations.append(sim.get_sensor_observations())
            step += 1. / 60.

        locobot.velocity_control.angular_velocity = np.array([0., 0., 0.])

        v, w = locobot.rotation.vector, locobot.rotation.scalar
        current_euler = Rotation.from_quat([v.x, v.y, v.z, w]).as_euler('xyz')
    else:
        return angular_vel_interp

    return observations


def move_to_point(sim, locobot, goal, args, dt=1, step=None):
    start = np.array(locobot.translation)
    end = np.array(goal)

    diff = end - start
    a = 4 * diff / (dt ** 2)

    def vel_interp(t):
        if t < dt / 2:
            return a * t
        else:
            return -a * t + a * dt

    start_time = sim.get_world_time()
    observations = []
    if step is None:
        step = 0
        while sim.get_world_time() < start_time + dt:
            locobot.velocity_control.linear_velocity = vel_interp(step)
            sim.step_physics(1.0 / 60.0)
            if args.make_video:
                observations.append(sim.get_sensor_observations())
            step += 1. / 60.
        locobot.velocity_control.linear_velocity = np.array([0., 0., 0.])
    else:
        return vel_interp

    return observations


def init_locobot(locobot, start):
    locobot.translation = start

    vel_control: habitat_sim.physics.VelocityControl = locobot.velocity_control
    vel_control.linear_velocity = [0.0, 0.0, 0.0]
    vel_control.angular_velocity = [0.0, 0.0, 0.0]

    # simulate robot dropping into place
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = False
    vel_control.ang_vel_is_local = False

    locobot.motion_type = habitat_sim.physics.MotionType.KINEMATIC


def run_locobot_collision_sp(sim, locobot1, locobot2, args, dt_per_action=1):
    start = np.zeros(3)
    goal = sim.pathfinder.get_random_navigable_point()

    init_locobot(locobot1, start)
    init_locobot(locobot2, goal)

    l1points = get_shortest_path(sim, locobot1.translation, goal)
    l2points = get_shortest_path(sim, locobot2.translation, start)

    observations = []
    for i in range(1, max(len(l1points), len(l2points))):
        # first perform rotations
        l1rot_f = rotate_to_face_point(sim, locobot1, l1points[i], args, dt=dt_per_action, step=1)
        l2rot_f = rotate_to_face_point(sim, locobot2, l2points[i], args, dt=dt_per_action, step=1)
        for step in np.arange(0, 1 + 1 / 60, 1 / 60):
            locobot1.velocity_control.angular_velocity = l1rot_f(step)
            locobot2.velocity_control.angular_velocity = l2rot_f(step)
            sim.step_physics(1.0 / 60.0)
            if args.make_video:
                observations.append(sim.get_sensor_observations())
        locobot1.velocity_control.angular_velocity = np.zeros(3)
        locobot2.velocity_control.angular_velocity = np.zeros(3)

        # second perform translation
        l1lin_f = move_to_point(sim, locobot1, l1points[i], args, dt=dt_per_action, step=1)
        l2lin_f = move_to_point(sim, locobot2, l2points[i], args, dt=dt_per_action, step=1)
        for step in np.arange(0, 1 + 1 / 60, 1 / 60):
            # check for collision
            if np.linalg.norm(locobot1.translation - locobot2.translation) <= 0.5:
                print('Collision!')
                return observations
            locobot1.velocity_control.linear_velocity = l1lin_f(step)
            locobot2.velocity_control.linear_velocity = l2lin_f(step)
            sim.step_physics(1.0 / 60.0)
            if args.make_video:
                observations.append(sim.get_sensor_observations())
        locobot1.velocity_control.linear_velocity = np.zeros(3)
        locobot2.velocity_control.linear_velocity = np.zeros(3)

    l1rot_f = rotate_to_face_point(sim, locobot1, l1points[0], args, dt=dt_per_action, step=1)
    l2rot_f = rotate_to_face_point(sim, locobot2, l2points[0], args, dt=dt_per_action, step=1)
    for step in np.arange(0, 1 + 1 / 60, 1 / 60):
        locobot1.velocity_control.angular_velocity = l1rot_f(step)
        locobot2.velocity_control.angular_velocity = l2rot_f(step)
        sim.step_physics(1.0 / 60.0)
        if args.make_video:
            observations.append(sim.get_sensor_observations())
    locobot1.velocity_control.angular_velocity = np.zeros(3)
    locobot2.velocity_control.angular_velocity = np.zeros(3)

    return observations


def run_locobot_shortest_path(sim: habitat_sim.Simulator, locobot: ManagedRigidObject, args: Arguments):
    locobot.translation = [0., 0., 0.]

    vel_control: habitat_sim.physics.VelocityControl = locobot.velocity_control
    vel_control.linear_velocity = [0.0, 0.0, 0.0]
    vel_control.angular_velocity = [0.0, 0.0, 0.0]

    # simulate robot dropping into place
    observations = simulate(sim, dt=1.5, get_frames=args.make_video)
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = False
    vel_control.ang_vel_is_local = False

    # pick random point until it is possible to move there
    goal = sim.pathfinder.get_random_navigable_point()

    points = get_shortest_path(sim, locobot.translation, goal)
    # observations += rotate_to_face_point(sim, locobot, [1, 0, 0], args)
    # observations += move_to_point(sim, locobot, [1, 0, 0], args)
    for point in points[1:]:
        observations += rotate_to_face_point(sim, locobot, point, args)
        observations += move_to_point(sim, locobot, point, args)

    observations += rotate_to_face_point(sim, locobot, [0, 0, 0], args)
    return observations


def run_locobot_example(sim, locobot: ManagedRigidObject, args: Arguments):
    # load the lobot_merged asset

    locobot.translation = [1.75, -1.02, 0.4]

    vel_control = locobot.velocity_control
    vel_control.linear_velocity = [0.0, 0.0, -1.0]
    vel_control.angular_velocity = [0.0, 2.0, 0.0]

    # simulate robot dropping into place
    observations = simulate(sim, dt=1.5, get_frames=args.make_video)

    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.ang_vel_is_local = True

    # simulate forward and turn
    observations += simulate(sim, dt=1.0, get_frames=args.make_video)

    vel_control.controlling_lin_vel = False
    vel_control.angular_velocity = [0.0, 1.0, 0.0]

    # simulate turn only
    observations += simulate(sim, dt=1.5, get_frames=args.make_video)

    vel_control.angular_velocity = [0.0, 0.0, 0.0]
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True

    # simulate forward only with damped angular velocity (reset angular velocity to 0 after each step)
    observations += simulate(sim, dt=1.0, get_frames=args.make_video)

    vel_control.angular_velocity = [0.0, -1.25, 0.0]

    # simulate forward and turn
    observations += simulate(sim, dt=2.0, get_frames=args.make_video)

    vel_control.controlling_ang_vel = False
    vel_control.controlling_lin_vel = False

    # simulate settling
    observations += simulate(sim, dt=3.0, get_frames=args.make_video)

    return observations


def main():
    args: Arguments = parse_args(Arguments)

    if "google.colab" in sys.modules:
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

    # create the simulators AND resets the simulator

    cfg = make_configuration(args)
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    agent_transform = place_agent(sim)

    # get the primitive assets attributes manager
    prim_templates_mgr = sim.get_asset_template_manager()

    # get the physics object attributes manager
    obj_templates_mgr = sim.get_object_template_manager()

    # get the rigid object manager
    rigid_obj_mgr = sim.get_rigid_object_manager()

    locobot = load_locobot(sim, obj_templates_mgr, rigid_obj_mgr, args, None)
    locobot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    locobot2 = load_locobot(sim, obj_templates_mgr, rigid_obj_mgr, args, None)

    observations = run_locobot_collision_sp(sim, locobot, locobot2, args)

    # remove the agent's body while preserving the SceneNode
    rigid_obj_mgr.remove_object_by_id(locobot.object_id, delete_object_node=False)

    # demonstrate that the locobot object does not now exist'
    print("Locobot is still alive : {}".format(locobot.is_alive))

    # video rendering with embedded 1st person view
    if args.make_video:
        vut.make_video(
            observations,
            "birdseye",
            "color",
            args.output_path + "collision",
            open_vid=args.show_video,
        )


if __name__ == '__main__':
    main()
