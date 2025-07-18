import xml.etree.ElementTree as ET
import mink
import numpy as np


def configuration_reached(
    configuration: mink.Configuration,
    task: mink.Task,
    pos_threshold: float = 5e-3,
    ori_threshold: float = 5e-3,
) -> float:
    """Check if a desired configuration has been reached."""
    err = task.compute_error(configuration)
    return (
        np.linalg.norm(err[:3]) <= pos_threshold
        and np.linalg.norm(err[3:]) <= ori_threshold
    )


def add_fromto_sensors(xml_path, target_geoms, goal_geoms, cutoff=1.0):
    """
    Clears any existing <fromto> sensors and adds new ones for all (target, goal) pairs to the <sensor> section of a MuJoCo XML file.
    NOTE: This also changes the formatting of the scene.xml, pay attention to not commit these changes.

    Args:
        xml_path (str): Path to the MuJoCo XML file.
        target_geoms (list of str): List of target geometry names.
        goal_geoms (list of str): List of goal geometry names.
        cutoff (float): Cutoff distance for the sensor.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")
    else:
        # Remove all existing children (clear all sensors)
        for child in list(sensor_elem):
            sensor_elem.remove(child)

    for geom1 in target_geoms:
        for geom2 in goal_geoms:
            fromto_elem = ET.Element("fromto")
            fromto_elem.set("geom1", geom1)
            fromto_elem.set("geom2", geom2)
            fromto_elem.set("cutoff", str(cutoff))
            sensor_elem.append(fromto_elem)

    tree.write(xml_path)
