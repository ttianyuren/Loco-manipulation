import numpy as np
import genesis as gs
import xml.etree.ElementTree as ET

urdf_path = "model/tiago/tiago_dual.urdf"

tree = ET.parse(urdf_path)
root = tree.getroot()

controlled_joint_names = [
    "torso_lift_joint",
    
    "arm_left_1_joint", 
    "arm_left_2_joint", 
    "arm_left_3_joint",
    "arm_left_4_joint", 
    "arm_left_5_joint", 
    "arm_left_6_joint",
    "arm_left_7_joint",
    "arm_right_1_joint", 
    "arm_right_2_joint", 
    "arm_right_3_joint",
    "arm_right_4_joint", 
    "arm_right_5_joint", 
    "arm_right_6_joint",
    "arm_right_7_joint",
    
    "wheel_left_joint", 
    "wheel_right_joint"
]

gs.init(backend=gs.cpu)


scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
tiago = scene.add_entity(gs.morphs.URDF(file=urdf_path))

scene.build()

dofs_idx = []

for joint_name in controlled_joint_names:
    joint = tiago.get_joint(joint_name)
    if joint is None:
        print(f" Joint '{joint_name}'")
    else:
        joint_idx = joint.dof_idx_local
        dofs_idx.append(joint_idx)
        print(f" Joint: {joint_name} - IDX: {joint_idx}")

kp_values = np.array([
    1000,  # **Torso**
    
    # **left arm**
    4500, 4500, 3500, 3500, 2000, 2000, 2000,
    
    # **right arm**
    4500, 4500, 3500, 3500, 2000, 2000, 2000,

    # **base**
    500, 500
])
kv_values = np.array([
    100,  # **Torso**

    # **left arm**
    450, 450, 350, 350, 200, 200, 200,

    # **right arm**
    450, 450, 350, 350, 200, 200, 200,

    # **base**
    50, 50
])
force_limits = np.array([
    2000,  # **Torso**

    # **left arm**
    43, 43, 26, 26, 3, 6.6, 6.6,

    # **right arm**
    43, 43, 26, 26, 3, 6.6, 6.6,

    # **base**
    600, 600
])



tiago.set_dofs_kp(kp=kp_values, dofs_idx_local=dofs_idx)
tiago.set_dofs_kv(kv=kv_values, dofs_idx_local=dofs_idx)
tiago.set_dofs_force_range(lower=-force_limits, upper=force_limits, dofs_idx_local=dofs_idx)

for i in range(1000):
    if i == 0:
        print("➡ reset")
        tiago.control_dofs_position(np.zeros(len(dofs_idx)), dofs_idx)

    elif i == 250:
        print("➡ Forward kinematic ")
        target_positions = np.array([0.5] * (len(dofs_idx) - 2)) 
        tiago.control_dofs_position(target_positions, dofs_idx[:-2])

    elif i == 300:
        print("➡ Torso Up")
        tiago.control_dofs_position(
            np.array([0.3] + [0] * (len(dofs_idx) - 1)),  
            dofs_idx
        )

    elif i == 500:
        print("➡ Torso down")
        tiago.control_dofs_position(
            np.array([0.0] + [0] * (len(dofs_idx) - 1)),
            dofs_idx
        )

    elif i == 750:
        print("➡ Base runing ")
        tiago.control_dofs_velocity(
            np.array([5.0, 2.0]),  # 5 2 rad/s
            dofs_idx[-2:]  # 
        )

    elif i == 100000:
        print("➡ Stop")
        tiago.control_dofs_velocity(
            np.array([0.0, 0.0]),  
            dofs_idx[-2:]
        )

    scene.step()  # 
