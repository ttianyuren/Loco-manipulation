# Differential IK Servoing in MuJoCo

Differential inverse kinematics (IK) servoing for whole-body teleoperation in MuJoCo, using [mink](https://github.com/kevinzakka/mink).

## Setup

Create and activate a virtual environment using Conda:

```bash
conda create -n mink_servo python=3.12
conda activate mink_servo
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dual-Arm Tiago Example

To launch the dual-arm Tiago simulation (24 degrees of freedom), run:

```bash
python3 sim/tiago_dual.py
```

## Controlling the Robot

### Keyboard

Keyboard-based control of the end effector is implemented using the [keyboard module](https://github.com/kevinzakka/mink/tree/main/mink/contrib/keyboard_teleop) from Mink. The key mappings can be found in the corresponding [KEYBOARD.md](https://github.com/kevinzakka/mink/blob/main/mink/contrib/keyboard_teleop/KEYBOARD.md) file.

### MoCAPs

The robot can also be controlled using [MoCap bodies](https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies), which allow interaction via mouse input. To use them:

1. Double-click the desired MoCap box (blue or red) in the MuJoCo scene to select it.
2. Then use the following mouse + keyboard combinations:

   - `Ctrl + Right Mouse Button (Hold)`: Translate the MoCap (position control)  
   - `Ctrl + Left Mouse Button (Hold)`: Rotate the MoCap (orientation control)

This method provides an intuitive way to guide the robot by manipulating reference bodies directly in the scene.

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
