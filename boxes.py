import mujoco
import mujoco.viewer
from pathlib import Path
from mink import CollisionAvoidanceLimit
import numpy as np

_HERE = Path(__file__).parent
_XML = _HERE / "models" / "minimal_env" / "two_boxes.xml"

# Make sure the file exists
if not _XML.exists():
    raise FileNotFoundError(f"{_XML} does not exist.")

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path(str(_XML))

# Create a data object for the model
data = mujoco.MjData(model)

box_1_geom_id = model.geom("box1").id
box_2_geom_id = model.geom("box2").id

limit = CollisionAvoidanceLimit(
    model=model,
    geom_pairs=[],
    collision_detection_distance=0.05
)

# Launch a viewer
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    while viewer.is_running():
        max_dist = 0.05
        fromto = np.zeros(6)
        dist = mujoco.mj_geomDistance(
            model, data, box_1_geom_id, box_2_geom_id, max_dist, fromto
        )
        print(f"Distance between box1 and box2: {dist} with collision detection distance {max_dist}")