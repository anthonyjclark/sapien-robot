"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by what agents/actors are
loaded, how agents/actors are randomly initialized during env resets, how goals are randomized and parameterized in observations, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the poses of all actors, articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do. If followed correctly you can easily build a
task that can simulate on the CPU and be parallelized on the GPU without having to manage GPU memory and parallelization apart from some
code that need to be written in batched mode (e.g. reward, success conditions)

For a minimal implementation of a simple task, check out
mani_skill /envs/tasks/push_cube.py which is annotated with comments to explain how it is implemented
"""

from typing import Any, Union

import numpy as np
import sapien
import torch

# from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils

# from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from twsim.robots.transwheel import TransWheel


# TODO: set a reasonable number of max episode steps
@register_env("Plane-v1", max_episode_steps=200)
class Plane(BaseEnv):
    """This is a flat plane environment for debugging purposes.

    The success condition is computed by reaching a goal position.
    """

    SUPPORTED_ROBOTS = ["transwheel"]
    agent: TransWheel

    def __init__(
        self, *args, robot_uids="transwheel", robot_init_qpos_noise=0.02, **kwargs
    ):
        # TODO: initial position noise is not used
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # TODO: let user pass in target position/pose
        self.target_pose = sapien.Pose(p=[0.5, 0, 0.5], q=[1, 0, 0, 0])
        self.target_radius = 0.05
        self.ground_threshold = -0.1

        # TODO: set an initial position for the robot
        self.initial_pose = sapien.Pose()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: dict):
        "Set the robot's initial pose."
        super()._load_agent(options, self.initial_pose)

    def _load_scene(self, options: dict):
        "Construct the scene ManiSkill will automatically create actors in every sub-scene)."
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        plane_half_size = (10, 1.5e-1, 1e-2)
        plane_color = (0.2, 0.2, 0.2)

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[3e-1, 0, 0], q=[1, 0, 0, 0])  # type: ignore
        builder.add_box_collision(half_size=plane_half_size)
        builder.add_box_visual(half_size=plane_half_size, material=plane_color)
        builder.build_static(name="ground-plane")

    # @property
    # def _default_sensor_configs(self):
    #     # To customize the sensors that capture images/point clouds for the environment observations,
    #     # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
    #     pose = sapien_utils.look_at(
    #         eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
    #     )  # sapien_utils.look_at is a utility to get the pose of a camera that looks at a target

    #     # to see what all the sensors capture in the environment for observations, run env.render_sensors() which returns an rgb array you can visualize
    #     return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        # TODO: look into other camera options (specifically the entity uid and mount)
        return [CameraConfig("render_camera", pose, width=512, height=512, fov=1)]

    # def _setup_sensors(self, options: dict):
    #     # default code here will setup all sensors. You can add additional code to change the sensors e.g.
    #     # if you want to randomize camera positions
    #     return super()._setup_sensors()

    # def _load_lighting(self, options: dict):
    #     # default code here will setup all lighting. You can add additional code to change the lighting e.g.
    #     # if you want to randomize lighting in the scene
    #     return super()._load_lighting()

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        "Set states for all non-static objects, including the robot."
        self.agent.robot.set_pose(self.initial_pose)

        # NOTE: here is where we would randomly set new poses for the environments (steps or terrain)
        # https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html#episode-initialization-randomization

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def evaluate(self):
        "(Batched) Return success and failure conditions for the task."

        # this function is used primarily to determine success and failure of a task, both of which are optional. If a dictionary is returned
        # containing "success": bool array indicating if the env is in success state or not, that is used as the terminated variable returned by
        # self.step. Likewise if it contains "fail": bool array indicating the opposite (failure state or not) the same occurs. If both are given
        # then a logical OR is taken so terminated = success | fail. If neither are given, terminated is always all False.
        #
        # You may also include additional keys which will populate the info object returned by self.step and that will be given to
        # `_get_obs_extra` and `_compute_dense_reward`. Note that as everything is batched, you must return a batched array of
        # `self.num_envs` booleans (or 0/1 values) for success and fail as done in the example below

        robot_position = self.agent.robot.get_pose().get_p()
        print(f"{robot_position.shape=}")
        print(f"{self.target_pose.p.shape=}")

        distance = np.linalg.norm((robot_position.cpu() - self.target_pose.p))
        print(f"{distance.shape=}")

        # Robot is at target position
        success = torch.tensor(
            [distance < self.target_radius], device=self.device, dtype=torch.bool
        )
        print(f"{success=}")

        # Robot has fallen off of the ground plane
        fail = torch.tensor(
            [10 < self.ground_threshold],
            device=self.device,
            dtype=torch.bool,
        )
        print(f"{fail=}")

        return {"success": success, "fail": fail}

    def _get_obs_extra(self, info: dict):
        # should return an dict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state". Moreover, you have access to the info object
        # which is generated by the `evaluate` function above
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of of self.num_envs rewards as done in the example below.
        # Moreover, you have access to the info object which is generated by the `evaluate` function above
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    def set_state_dict(self, state):
        # this function complements get_state and sets any non simulation state related data correctly so the environment behaves
        # the exact same in terms of output rewards, observations, success etc. should you reset state to a given state and take the same actions
        self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)
