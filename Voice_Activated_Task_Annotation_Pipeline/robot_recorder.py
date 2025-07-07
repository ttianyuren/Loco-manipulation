"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import keyboard

class RobotRecorder:
    def __init__(self, config):
        self.config = config
        self.directory = config["directory"]
        os.makedirs(self.directory, exist_ok=True)

        self.environment = config["environment"]
        self.robots = config["robots"]
        self.config_name = config.get("config", "default")
        self.arm = config.get("arm", "right")
        self.camera = config.get("camera", "agentview")
        self.controller = config.get("controller", None)
        self.device = config.get("device", "keyboard")
        self.pos_sensitivity = config.get("pos_sensitivity", 1.0)
        self.rot_sensitivity = config.get("rot_sensitivity", 1.0)
        self.renderer = config.get("renderer", "mjviewer")
        self.max_fr = config.get("max_fr", 20)
        self.reverse_xy = config.get("reverse_xy", False)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tmp_directory = os.path.join(self.directory, "tmp", timestamp)
        os.makedirs(self.tmp_directory, exist_ok=True)

    def collect_human_trajectory(self, env, device, arm, max_fr):
        """
        Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
        The rollout trajectory is saved to files in npz format.
        Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

        Args:
            env (MujocoEnv): environment to control
            device (Device): to receive controls from the device
            arms (str): which arm to control (eg bimanual) 'right' or 'left'
            max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
        """

        env.reset()
        env.render()

        task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
        device.start_control()

        for robot in env.robots:
            robot.print_action_info_dict()

        # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        # Loop until we get a reset from the input or the task completes
        while True:
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            env.step(env_action)
            env.render()

            # Also break if we complete the task
            if task_completion_hold_count == 0:
                break

            # state machine to check for having a success for 10 consecutive timesteps
            if env._check_success():
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1  # latched state, decrement count
                else:
                    task_completion_hold_count = 10  # reset count on first success timestep
            else:
                task_completion_hold_count = -1  # null the counter if there's no success

            # limit frame rate if necessary
            if max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)

        # cleanup for end of data collection episodes
        env.close()


    def gather_demonstrations_as_hdf5(self, directory, out_dir, env_info):
        """
        Gathers the demonstrations saved in @directory into a
        single hdf5 file.

        The strucure of the hdf5 file is as follows.

        data (group)
            date (attribute) - date of collection
            time (attribute) - time of collection
            repository_version (attribute) - repository version used during collection
            env (attribute) - environment name on which demos were collected

            demo1 (group) - every demonstration has a group
                model_file (attribute) - model xml string for demonstration
                states (dataset) - flattened mujoco states
                actions (dataset) - actions applied during demonstration

            demo2 (group)
            ...

        Args:
            directory (str): Path to the directory containing raw demonstrations.
            out_dir (str): Path to where to store the hdf5 file.
            env_info (str): JSON-encoded string containing environment information,
                including controller and robot info
        """

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        hdf5_path = os.path.join(out_dir, f"{timestamp}.hdf5")
        f = h5py.File(hdf5_path, "w")

        # store some metadata in the attributes of one group
        grp = f.create_group("data")

        num_eps = 0
        env_name = None  # will get populated at some point

        for ep_directory in os.listdir(directory):
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            states = []
            actions = []
            success = False

            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or dic["successful"]

            if len(states) == 0:
                continue

            # Add only the successful demonstration to dataset
            if success:
                print("Demonstration is successful and has been saved")
                # Delete the last state. This is because when the DataCollector wrapper
                # recorded the states and actions, the states were recorded AFTER playing that action,
                # so we end up with an extra state at the end.
                del states[-1]
                assert len(states) == len(actions)

                num_eps += 1
                ep_data_grp = grp.create_group("demo_{}".format(num_eps))

                # store model xml as an attribute
                xml_path = os.path.join(directory, ep_directory, "model.xml")
                with open(xml_path, "r") as f:
                    xml_str = f.read()
                ep_data_grp.attrs["model_file"] = xml_str

                # write datasets for states and actions
                ep_data_grp.create_dataset("states", data=np.array(states))
                ep_data_grp.create_dataset("actions", data=np.array(actions))
            else:
                print("Demonstration is unsuccessful and has NOT been saved")

        # write dataset attributes (metadata)
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info

        f.close()

    def collect_trajectory(self):
            # Get controller config
        controller_config = load_composite_controller_config(
            controller=self.controller,
            robot=self.robots[0],
        )

        # if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        #     # mink-speicific import. requires installing mink
        #     from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

        # Create argument configuration
        env_config = {
            "env_name": self.environment,
            "robots": self.robots,
            "controller_configs": controller_config,
        }

        # Check if we're using a multi-armed environment and use env_configuration argument if so
        if "TwoArm" in self.environment:
            env_config["env_configuration"] = self.config_name

        # Create environment
        env = suite.make(
            **env_config,
            has_renderer=True,
            renderer=self.renderer,
            has_offscreen_renderer=False,
            render_camera=self.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

        # Wrap this with visualization wrapper
        env = VisualizationWrapper(env)

        # Grab reference to controller config and convert it to json-encoded string
        env_info = json.dumps(env_config)

        # wrap the environment with data collection wrapper
        env = DataCollectionWrapper(env, self.tmp_directory)

        # initialize device
        if self.device == "keyboard":
            from robosuite.devices import Keyboard

            device = Keyboard(
                env=env,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        elif self.device == "spacemouse":
            from robosuite.devices import SpaceMouse

            device = SpaceMouse(
                env=env,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        elif self.device == "dualsense":
            from robosuite.devices import DualSense

            device = DualSense(
                env=env,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
                reverse_xy=self.reverse_xy,
            )
        elif self.device == "mjgui":
            assert self.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
            from robosuite.devices.mjgui import MJGUI

            device = MJGUI(env=env)
        else:
            raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # new_dir = os.path.join(self.directory, timestamp)
        # os.makedirs(new_dir, exist_ok=True)

        # collect demonstrations
        while True:
            self.collect_human_trajectory(env, device, self.arm, self.max_fr)
            self.gather_demonstrations_as_hdf5(self.tmp_directory, self.directory, env_info) # tmp_directory saves time and trajectory，new_dir is the path of h5py
            print(f"Recording finished, file is saved under: {self.directory}")
            break
        # return new_dir

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="./trajectory_storage"),
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    args = parser.parse_args()

    config = vars(args) #change argparse.Namespace into directionary
    collector = RobotRecorder(config)
    collector.collect_trajectory()