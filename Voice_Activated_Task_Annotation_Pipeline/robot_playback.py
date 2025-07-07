import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite

import glob

class RobotPlayback:
    def find_latest_hdf5(self, folder):
        hdf5_files = glob.glob(os.path.join(folder, '*.hdf5')) #return a folder list
        if not hdf5_files:
            raise FileNotFoundError(f"No hdf5 files found in {folder}.")
        latest_file = max(hdf5_files, key=os.path.getctime) #could return the latest established .hdf5 file
        return latest_file
    
    def playback_demo(self, hdf5_path, use_actions=True):
        print(f"Using hdf5 file: {hdf5_path}")
        
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"{hdf5_path} does not exist.")

        f = h5py.File(hdf5_path, "r")
        env_name = f["data"].attrs["env"]
        env_info = json.loads(f["data"].attrs["env_info"])

        # print("env_name from hdf5:", env_name)
        # print("env_info from hdf5:", env_info)

        env = robosuite.make(
            **env_info,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

        # list of all demonstrations episodes
        demos = list(f["data"].keys())

        while True:
            print("Playing back random episode... (press ESC to quit)")

            # select an episode randomly
            ep = random.choice(demos)

            # read the model xml, using the metadata stored in the attribute for this episode
            model_xml = f["data/{}".format(ep)].attrs["model_file"]

            env.reset()
            xml = env.edit_model_xml(model_xml)
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.viewer.set_camera(0)

            # load the flattened mujoco states
            states = f["data/{}/states".format(ep)][()]

            if use_actions:

                # load the initial state
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()

                # load the actions and play them back open-loop
                actions = np.array(f["data/{}/actions".format(ep)][()])
                num_actions = actions.shape[0]

                for j, action in enumerate(actions):
                    env.step(action)
                    env.render()

                    if j < num_actions - 1:
                        # ensure that the actions deterministically lead to the same recorded states
                        state_playback = env.sim.get_state().flatten()
                        if not np.all(np.equal(states[j + 1], state_playback)):
                            err = np.linalg.norm(states[j + 1] - state_playback)
                            print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

            else:

                # force the sequence of internal mujoco states one by one
                for state in states:
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    if env.renderer == "mjviewer":
                        env.viewer.update()
                    env.render()

        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="trajectory_storage",
        help="Path to your demonstration folder that contains .hdf5 files"
    )
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="Whether to use actions for playback"
    )
    args = parser.parse_args()
    
    robot_playback = RobotPlayback()
    try:
        hdf5_path = robot_playback.find_latest_hdf5(args.folder)
        robot_playback.playback_demo(hdf5_path, use_actions=args.use_actions)
    except FileNotFoundError as e:
        print(e)