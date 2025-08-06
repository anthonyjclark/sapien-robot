# PPO Experiments

For deployment:

    checkpoint: str                          # Path to the checkpoint file
    env_id: str = "PlaneVel-v1"              # Environment ID
    control_mode: str = "wheel_vel_ext_pos"  # Control mode
    cuda: bool = True                        # Use GPU for evaluation


~~~bash
# See the main README for instructions on how to install the environment
# Download checkpoints from our shared drive
python deploy.py --checkpoint CHECKPOINT_FILE --env-id SensorVel-v1
~~~
