import os

import habitat_sim
import numpy as np
from _magnum import Vector3
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat_sim import NavMeshSettings


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


def init_objects(sim):
    # Manager of Object Attributes Templates
    obj_attr_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()

    obj_attr_mgr.load_configs(
        str(os.path.join('data/', "replica_cad/configs/objects/"))
    )

    # Add a chair into the scene.
    #obj_path = "objects/ycb/configs/002_master_chef_can"
    obj_path = "replica_cad/configs/objects/frl_apartment_wall_cabinet_01"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join('data/', obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_id(chair_template_id)
    obj_attr_mgr.register_template(chair_attr)

    # Object's initial position 3m away from the agent.
    object = rigid_obj_mgr.add_object_by_template_id(chair_template_id)
    set_object_in_front_of_agent(sim, object, z_offset=1.75, x_offset=0.35)
    object.motion_type = habitat_sim.physics.MotionType.STATIC

    # Object's final position 7m away from the agent
    goal = rigid_obj_mgr.add_object_by_template_id(chair_template_id)
    set_object_in_front_of_agent(sim, goal, z_offset=2.5, x_offset=-0.35)
    goal.motion_type = habitat_sim.physics.MotionType.STATIC

    return object, goal

@registry.register_task(name="StaticObstacleNav-v0")
class NewNavigationTask(NavigationTask):


    def __init__(self, config, sim, dataset):
        logger.info("Creating a new type of task")
        super().__init__(config=config, sim=sim, dataset=dataset)
        init_objects(sim)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = sim.agents[0].agent_config.radius
        self.navmesh_settings.agent_height = sim.agents[0].agent_config.height

        sim.recompute_navmesh(sim.pathfinder, self.navmesh_settings, True)


    def _check_episode_is_active(self, *args, **kwargs):
        logger.info(
            "Current agent position: {}".format(self._sim.get_agent_state())
        )
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
        print('here!')
