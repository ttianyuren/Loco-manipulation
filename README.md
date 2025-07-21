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
python3 sim/tiago_dual.py
```

## Keyboard Control

Keyboard-based control of the end effector is currently implemented using the [keyboard module](https://github.com/kevinzakka/mink/tree/main/mink/contrib/keyboard_teleop) from Mink. The key mappings can be found in the corresponding [KEYBOARD.md](https://github.com/kevinzakka/mink/blob/main/mink/contrib/keyboard_teleop/KEYBOARD.md) file.

## Robot Model

The MuJoCo model for the dual-arm Tiago robot is sourced from the [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago_dual) repository.  For details on modifications or updates made to the model, please refer to the corresponding [CHANGELOG.md](./models/pal_tiago_dual/CHANGELOG.md).

## Configuration Update

There are two methods for updating the configuration of the robot:

1. **Direct joint configuration**  
   The joint positions are set explicitly, bypassing actuator dynamics.

2. **Configuration through actuator dynamics**  
   This approach incorporates the dynamics of the actuators. The appropriate control law depends on the actuator type. Currently, we use **position-controlled actuators** with a **Proportional-Derivative (PD) controller**, defined by: $\boldsymbol{\tau} = K_p (\boldsymbol{q}_d - \boldsymbol{q}) - K_d \, \dot{\boldsymbol{q}}$

Depending on your chosen control method, comment out the appropriate line in [`sim/tiago_dual.py`](./sim/tiago_dual.py):

```python
data.qpos[:] = np.array(configuration.q, dtype=float)  # Set configuration directly
data.ctrl[actuator_ids] = configuration.q[dof_ids]     # With actuator dynamics
```

## Code Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for code formatting. To format, run:

```bash
ruff format [file]
```
