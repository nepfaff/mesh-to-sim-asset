import argparse
import os
import re

from lxml import etree as ET
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import DiagramBuilder, Simulator, StartMeshcat


def get_link_name_from_sdf(sdf_file_path: str) -> str:
    """Extract the appropriate link name from an SDF file.

    Logic:
    1. If there's only one link, use that
    2. If there are multiple links and one contains "body", use that
    3. Otherwise, use the first one

    Args:
        sdf_file_path: Path to the SDF file

    Returns:
        The selected link name
    """
    with open(sdf_file_path, "r") as file:
        sdf_content = file.read()

    # Remove XML comments to avoid matching content inside them.
    sdf_content_no_comments = re.sub(r"<!--.*?-->", "", sdf_content, flags=re.DOTALL)

    try:
        root = ET.fromstring(sdf_content_no_comments)
    except ET.XMLSyntaxError as e:
        raise ValueError(f"Invalid SDF XML syntax in {sdf_file_path}: {e}")

    # Find all link elements and extract their names.
    link_elements = root.xpath(".//link[@name]")
    link_names = [link.get("name") for link in link_elements]

    if not link_names:
        raise ValueError(f"No links found in SDF file: {sdf_file_path}")

    # If there's only one link, use it.
    if len(link_names) == 1:
        return link_names[0]

    # If there are multiple links, look for one containing "body".
    for link_name in link_names:
        if "body" in link_name.lower():
            return link_name

    # Otherwise, use the first one.
    return link_names[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sdf_file_path", type=str)
    parser.add_argument("--time_step", type=float, required=False, default=0.01)
    parser.add_argument(
        "--position",
        help="Position in meters.",
        type=str,
        required=False,
        default="0, 0, 0.1",
    )
    parser.add_argument(
        "--rotation",
        help="Rotation in degrees.",
        type=str,
        required=False,
        default="0, 0, 0",
    )
    parser.add_argument("--use_ramp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sdf_file_path = os.path.abspath(args.sdf_file_path)
    time_step = args.time_step
    position = args.position
    rotation = args.rotation
    use_ramp = args.use_ramp

    # Get the appropriate link name from the SDF file.
    link_name = get_link_name_from_sdf(sdf_file_path)
    print(f"Identified link name: {link_name}")

    meshcat = StartMeshcat()

    # Create and load the scenario.
    floor_sdf_path = os.path.abspath("models/floor.sdf")
    ramp_sdf_path = os.path.abspath("models/ramp.sdf")
    ramp_model_directive = (
        f"""
        - add_model:
            name: ramp
            file: file://{ramp_sdf_path}
        - add_weld:
            parent: world
            child: ramp
    """
        if use_ramp
        else ""
    )
    scenario_data = f"""
        directives:
        - add_model:
            name: table
            file: file://{floor_sdf_path}
        - add_weld:
            parent: world
            child: floor
        - add_model:
            name: asset
            file: file://{sdf_file_path}
            default_free_body_pose:
                {link_name}:
                    translation: [{position}]
                    rotation: !Rpy {{ deg: [{rotation}]}}
        {ramp_model_directive}
        plant_config:
            time_step: {time_step}
    """
    scenario = LoadScenario(data=scenario_data)

    # Create the diagram.
    builder = DiagramBuilder()
    builder.AddNamedSystem(
        "station", MakeHardwareStation(scenario=scenario, meshcat=meshcat)
    )
    diagram = builder.Build()

    # Simulate.
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    meshcat.StartRecording()
    simulator.AdvanceTo(5.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()


if __name__ == "__main__":
    main()
