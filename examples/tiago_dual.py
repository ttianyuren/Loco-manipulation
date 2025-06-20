from pathlib import Path
from typing import List, Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib import TeleopMocap

_HERE = Path(__file__).parent
_XML = _HERE / "pal_tiago_dual" / "scene.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "1_joint",
    "2_joint",
    "3_joint",
    "4_joint",
    "5_joint",
    "6_joint",
    "7_joint",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Compute forces to counteract gravity for the given subtrees.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        subtree_ids: List of subtree ids. A subtree is defined as the kinematic tree
            starting at the body and including all its descendants. Gravity
            compensation forces will be applied to all bodies in the subtree.
        qfrc_applied: Optional array to store the computed forces. If not provided,
            the applied forces in `data` are used.
    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Bodies for which to apply gravity compensation.
    lift_subtree_id = model.body("torso_lift_link").id

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names: List[str] = []
    velocity_limits: dict[str, float] = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"arm_{prefix}_{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    joint_names.extend(["base_x","base_y","base_th","torso_lift_joint"])
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(f"{name}_position").id for name in joint_names])

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

    # Enable collision avoidance between the following geoms.
    # l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    # r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    # l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    # r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    # frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    # collision_pairs = [
    #     (l_wrist_geoms, r_wrist_geoms),
    #     (l_geoms + r_geoms, frame_geoms + ["table"]),
    # ]
    # collision_avoidance_limit = mink.CollisionAvoidanceLimit(
    #     model=model,
    #     geom_pairs=collision_pairs,  # type: ignore
    #     minimum_distance_from_collisions=0.05,
    #     collision_detection_distance=0.1,
    # )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        # collision_avoidance_limit,
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
        mink.move_mocap_to_frame(model, data, "left_gripper_target", "left_gripper", "site")
        mink.move_mocap_to_frame(model, data, "right_gripper_target", "right_gripper", "site")

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
            compensate_gravity(model, data, [lift_subtree_id])
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
