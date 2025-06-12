from dataclasses import dataclass

import tyro


@dataclass
class Args:
    "Evaluate a trained policy."

    # fmt: off

    checkpoint: str                          # Path to the checkpoint file
    env_id: str = "PlaneVel-v1"              # Environment ID
    control_mode: str = "wheel_vel_ext_pos"  # Control mode
    cuda: bool = True                        # Use GPU for evaluation

    # fmt: on


if __name__ == "__main__":
    #
    #  Parse arguments
    #

    args = tyro.cli(Args)

    import gymnasium as gym
    import torch
    from agent import Agent
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

    import twsim.envs  # noqa: F401
    from twsim.robots import transwheel  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #
    # Create the environment
    # NOTE: we create the environment when deploying so that we can use the same
    # observation and action spaces as the training environment.
    #

    # TODO: ManiSkill has a new feature branch that will make this easier:
    # https://github.com/haosulab/ManiSkill/blob/sim2real-features
    # We'll want to use that functionality when it is released.

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
    )

    # Create the evaluation environment
    env = gym.make(args.env_id, **env_kwargs)  # type: ignore

    # Flatten action spaces if needed
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    #
    # Create the agent and load the trained policy
    #

    observation_shape = env.single_observation_space.shape  # type: ignore
    action_shape = env.single_action_space.shape  # type: ignore

    agent = Agent(observation_shape, action_shape).to(device)
    agent.load_state_dict(torch.load(args.checkpoint))
    agent.eval()

    #
    # Main loop
    #

    # NOTE: you can run these lines to get the observation shape
    # obs, _ = env.reset()
    # print(f"==>> obs: {obs}")
    # print(f"==>> obs: {obs.shape}")

    step = 0
    while True:
        # TODO: setup observation
        # NOTE: for the StepVelSensor environment --> obs: torch.Size([1, 19])
        #  0..< 4: wheel motor velocities
        #  4..< 8: extension motor velocities (position control--> relative to wheels)
        #  8..<12: body orientation (quaternion)
        # 12..<15: body angular velocity (x, y, z)
        # 15..<18: body linear velocity (x, y, z)
        # 18..<19: collision/no-collision (1.0/0.0)
        # NOTE: the other environments (e.g., PlaneVel) do not include the last observation
        obs = torch.randn(1, 19).to(device)

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
            step += 1

            # TODO: translate actions to the robot's physical commands
            # the action space is scaled to [-1, 1], you can see what these mean physically
            # by exploring the TransWheel._controller_configs method in transwheel.py
            # At the time I am writing this, the following code is used
            #   For the wheel velocities:
            #     wheel_radius = 1.5         # inches
            #     max_linear_velocity = 0.5  # inches / second
            #     max_angular_velocity = max_linear_velocity / wheel_radius
            #     lower=-max_angular_velocity,
            #     upper=max_angular_velocity,
            #   For the extension angles:
            #     lower=0,
            #     upper=np.pi,

        # TODO: change this to something more useful
        if step == 100:
            break

    #
    # Clean up
    #

    env.close()
