import os

import habitat_sim
import numpy as np
from _magnum import Vector3
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat_sim import NavMeshSettings, ShortestPath

def remove_all_objects(sim):
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)




def set_object_in_front_of_agent(sim, obj, z_offset=-1.5, x_offset=1.):
    r"""
    Adds an object in front of the agent at some distance.
    """
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([x_offset, 0, z_offset])
    )
    obj.translation = Vector3(obj_translation)

    obj_node = obj.root_scene_node
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.04
    y_translation = Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    obj.translation = y_translation + obj.translation


def init_objects(sim, points):
    # Manager of Object Attributes Templates
    obj_attr_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()

    obj_attr_mgr.load_configs(
        str(os.path.join('data/', "replica_cad/configs/objects/"))
    )

    # Add a chair into the scene.
    # obj_path = "objects/ycb/configs/002_master_chef_can"
    obj_path = "replica_cad/configs/objects/frl_apartment_wall_cabinet_01"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join('data/', obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_id(chair_template_id)
    obj_attr_mgr.register_template(chair_attr)

    objects = []
    for point in points:
        # Object's initial position 3m away from the agent.
        object = rigid_obj_mgr.add_object_by_template_id(chair_template_id)
        object.translation = Vector3(point)
        object.motion_type = habitat_sim.physics.MotionType.STATIC
        objects.append(object)
    return objects


def move_to_point(locobot, goal, dt=1):
    start = np.array(locobot.translation)
    end = np.array(goal)

    diff = end - start
    a = 4 * diff / (dt ** 2)

    def vel_interp(t):
        if t < dt / 2:
            return a * t
        else:
            return -a * t + a * dt

    return vel_interp


def get_shortest_path(sim: habitat_sim.Simulator, start, end):
    shortest_path = ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = end
    sim.pathfinder.find_path(shortest_path)
    return shortest_path.points


@registry.register_task(name="DynamicObstacleNav-v0")
class NewNavigationTask(NavigationTask):

    def init_objects(self, o1):
        o1.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        vel_control: habitat_sim.physics.VelocityControl = o1.velocity_control
        vel_control.linear_velocity = [0.0, 0.0, 0.]
        vel_control.angular_velocity = [0.0, 0.0, 0.0]

        # simulate robot dropping into place
        vel_control.controlling_lin_vel = True
        vel_control.controlling_ang_vel = True
        vel_control.lin_vel_is_local = False
        vel_control.ang_vel_is_local = False

    def move_object(self, i):
        point, step, direction = self.current_trajec[i]
        shortest_path = self.shortest_paths[i]
        o = self.objects[i]
        frames = 60
        if step == frames:
            # reset step and increment point
            step = 1
            point += 1
            if point == len(shortest_path) - 1:
                # print("SWITCHING DIRECTIONS")
                # print(o.translation)
                direction *= -1
                point = 0
            o.velocity_control.linear_velocity = Vector3([0, 0, 0])
        if direction == 1:
            func = move_to_point(o, shortest_path[point + 1], dt=(frames / 60))
        else:
            func = move_to_point(o, shortest_path[len(shortest_path) - point - 2], dt=(frames / 60))

        o.velocity_control.linear_velocity = Vector3(func(step / frames))
        self.current_trajec[i] = (point, step + 1, direction)

    def recompute_mesh(self):
        for o in self.objects:
            o.motion_type = habitat_sim.physics.MotionType.STATIC
        self._sim.recompute_navmesh(self._sim.pathfinder, self.navmesh_settings, True)
        for o in self.objects:
            o.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    def random_points(self, sim, n):
        points = []
        for i in range(n):
            points.append(sim.pathfinder.get_random_navigable_point())
        return points

    def __init__(self, config, sim, dataset):
        logger.info("Creating a new type of task Version: 2")
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.start_points = self.random_points(sim, 2)
        self.end_points = self.random_points(sim, 2)
        self.start_points = [[3.76204, 0.17669876, 0.72620916], [2.2625003, 0.17669876, 0.23297535]]
        self.end_points = [[1.6716383, 0.17669876, 0.7140454], [4.0453434, 0.17669876, 1.03202792]]
        self.objects = init_objects(sim, self.start_points)
        self.current_trajec = []
        self.shortest_paths = []

        for start, end in zip(self.start_points, self.end_points):
            sp = get_shortest_path(sim, start, end)
            self.shortest_paths.append(sp)
            self.current_trajec.append((0, 1, 1))  # point 0, step 0, direction forward

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = sim.agents[0].agent_config.radius
        self.navmesh_settings.agent_height = sim.agents[0].agent_config.height

        self.acceptable_goal_radius = 0.1

        sim.recompute_navmesh(sim.pathfinder, self.navmesh_settings, True)

        for o in self.objects:
            self.init_objects(o)

    def _check_episode_is_active(self, *args, **kwargs):
        # logger.info(
        #    "Current agent position: {}".format(self._sim.get_agent_state())
        # )
        collision = self._sim.previous_step_collided
        stop_called = not getattr(self, "is_stop_called", False)
        return collision or stop_called

    def _step_single_action(
            self,
            observations,
            action_name,
            action,
            episode,
            is_last_action=True,
    ):
        super()._step_single_action(observations, action_name, action, episode, is_last_action=is_last_action)
        for i in range(len(self.objects)):
            self.move_object(i)
        self.recompute_mesh()
        # print('here!')
