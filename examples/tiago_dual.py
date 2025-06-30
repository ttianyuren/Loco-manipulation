from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib import TeleopMocap

_HERE = Path(__file__).parent
_XML = _HERE / "pal_tiago_dual" / "scene.xml"

# Combined joint names and velocity limits for all controlled joints
ARM_JOINT_NAMES = [
    "1_joint",
    "2_joint",
    "3_joint",
    "4_joint",
    "5_joint",
    "6_joint",
    "7_joint",
]

CONTROLLED_JOINTS_AND_LIMITS = [
    *[
        (f"arm_left_{n}", l)
        for n, l in zip(ARM_JOINT_NAMES, [1.95, 1.95, 2.35, 2.35, 1.95, 1.76, 1.76])
    ],
    *[
        (f"arm_right_{n}", l)
        for n, l in zip(ARM_JOINT_NAMES, [1.95, 1.95, 2.35, 2.35, 1.95, 1.76, 1.76])
    ],
    ("torso_lift_joint", 0.07),
    ("base_x", 0.5),
    ("base_y", 0.5),
    ("base_th", 0.5),
]

CONTROLLED_JOINT_NAMES = [name for name, _ in CONTROLLED_JOINTS_AND_LIMITS]
VEL_LIMITS = dict(CONTROLLED_JOINTS_AND_LIMITS)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    dof_ids = np.array([model.joint(name).id for name in CONTROLLED_JOINT_NAMES])
    actuator_ids = np.array(
        [model.actuator(f"{name}_position").id for name in CONTROLLED_JOINT_NAMES]
    )

    configuration = mink.Configuration(model)

    tasks = [
        base_task := mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        ),
        l_ee_task := mink.RelativeFrameTask(
            frame_name="left_gripper",
            frame_type="site",
            root_name="base_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        r_ee_task := mink.RelativeFrameTask(
            frame_name="right_gripper",
            frame_type="site",
            root_name="base_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
    ]

    collision_pairs = [
        (["finger_left_1", "finger_left_2"], ["tabletop"]),
    ]   
    
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.1,
        collision_detection_distance=0.15,
    )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocities=VEL_LIMITS),
        collision_avoidance_limit,
    ]

    base_mid = model.body("base_target").mocapid[0]
    l_mid = model.body("left_gripper_target").mocapid[0]
    r_mid = model.body("right_gripper_target").mocapid[0]
    solver = "daqp"
    pos_threshold = 5e-3
    ori_threshold = 5e-3
    max_iters = 5

    # Initialize key_callback function.
    key_callback = TeleopMocap(data)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)
        base_task.set_target_from_configuration(configuration)
        assert base_task.transform_target_to_world is not None

        # Initialize mocap targets at the end-effector site.
        mink.move_mocap_to_frame(
            model, data, "left_gripper_target", "left_gripper", "site"
        )
        mink.move_mocap_to_frame(
            model, data, "right_gripper_target", "right_gripper", "site"
        )

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            base_pose = data.mocap_pos[base_mid].copy()
            base_pose[2] = 0
            data.mocap_pos[base_mid] = base_pose
            # base_rpy = mink.SO3(data.mocap_quat[base_mid]).as_rpy_radians()
            # data.mocap_quat[base_mid] = mink.SO3.from_z_radians(base_rpy.yaw).wxyz
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))

            l_ee_task.set_target(mink.SE3.from_mocap_id(data, l_mid))
            r_ee_task.set_target(mink.SE3.from_mocap_id(data, r_mid))

            # Continuously check for autonomous key movement.
            key_callback.auto_key_move()

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                l_err = l_ee_task.compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
                r_err = r_ee_task.compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
                if (
                    l_pos_achieved
                    and l_ori_achieved
                    and r_pos_achieved
                    and r_ori_achieved
                ):
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)# Update the viewer with the new data.

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
