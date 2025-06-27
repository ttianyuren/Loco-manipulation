# Whole Body Teleoperation for Dual-Arm Tiago

Whole-body teleoperation of a dual-arm Tiago robot equipped with a holonomic base. It uses [mink](https://github.com/kevinzakka/mink) for differential inverse kinematics (IK).

## Setup

Create and activate a virtual environment using Conda:

```bash
conda create -n tiago_teleop python=3.12
conda activate tiago_teleop
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Example

Run the teleoperation example with:

```bash
python3 examples/tiago_dual.py
```

## Keyboard Control

Keyboard-based control of the end effector is currently implemented using the [keyboard module](https://github.com/kevinzakka/mink/tree/main/mink/contrib/keyboard_teleop) from Mink. The key mappings can be found in the corresponding [KEYBOARD.md](https://github.com/kevinzakka/mink/blob/main/mink/contrib/keyboard_teleop/KEYBOARD.md) file.

## Robot Model

The MuJoCo model for the dual-arm Tiago robot is taken from the [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago_dual) repository.
