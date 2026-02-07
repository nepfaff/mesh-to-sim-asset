import json
import os
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import DiagramBuilder, Meshcat, Simulator, StartMeshcat, VectorLogSink

from mesh_to_sim_asset.sdformat import create_rigid_hydro_sdf_file
from mesh_to_sim_asset.sim_metrics import (
    orientation_considered_average_displacement_error,
    orientation_considered_final_displacement_error,
)


@dataclass
class SimulationResult:
    simulation_time: float
    """The amount of real time in seconds that the simulation took to run."""
    states: np.ndarray
    """The states of the simulation at each sample time of shape (N, 13)."""
    sample_times: np.ndarray
    """The times at which the states were sampled of shape (N,)."""


@dataclass
class EvalResult:
    sdf_file_paths: list[Path]
    """The SDF files that were evaluated. All metric lists follow the same list
    order."""
    orientation_considered_final_errors: list[float]
    """The final orientation considered displacement error for each SDF file."""
    orientation_considered_average_errors: list[float]
    """The average orientation considered displacement error for each SDF file."""
    simulation_times: list[float]
    """The simulation time for each SDF file."""

    def save_to_disk(self, output_path: Path) -> None:
        """Save the evaluation result to disk."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sdf_file_paths": [str(p) for p in self.sdf_file_paths],
                    "orientation_considered_final_errors": self.orientation_considered_final_errors,
                    "orientation_considered_average_errors": self.orientation_considered_average_errors,
                    "simulation_times": self.simulation_times,
                },
                f,
                indent=2,
            )


def _run_single_static_equilibirum_sim(
    sdf_file_path: Path,
    meshcat: Meshcat,
    recording_output_dir: Path | None = None,
) -> SimulationResult:
    """Run a single static equilibrium simulation. The object is dropped onto a flat
    surface from a height of 10cm and simulates for 5 seconds.

    Args:
        sdf_file_path (Path): The path to the SDF file to simulate.
        meshcat (Meshcat): The meshcat instance to use for visualization. The meshcat
            instance will be cleared before the simulation is run.
        recording_output_dir (Path | None): The directory to save the recording to.

    Returns:
        SimulationResult: The result of the simulation.
    """
    # Clear the meshcat.
    meshcat.Delete()

    # Create and load the scenario.
    floor_sdf_path = os.path.abspath("models/floor.sdf")
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
                {sdf_file_path.stem}_body_link:
                    translation: [0, 0, 0.1]
                    rotation: !Rpy {{ deg: [0, 0, 0]}}
        plant_config:
            time_step: 0.01
    """
    scenario = LoadScenario(data=scenario_data)

    # Create the diagram.
    builder = DiagramBuilder()
    station = builder.AddNamedSystem(
        "station", MakeHardwareStation(scenario=scenario, meshcat=meshcat)
    )
    # Add state logger.
    state_logger: VectorLogSink = builder.AddNamedSystem(
        "state_logger", VectorLogSink(13)
    )
    builder.Connect(station.GetOutputPort("asset_state"), state_logger.get_input_port())
    diagram = builder.Build()

    # Simulate and measure wall time.
    simulator = Simulator(diagram)
    simulator.Initialize()
    meshcat.StartRecording()
    start_time = time.time()
    simulator.AdvanceTo(5.0)
    end_time = time.time()
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # Get the states.
    state_log = state_logger.FindLog(context=simulator.get_context())
    states = state_log.data().T  # Shape (n_samples, 13)
    sample_times = state_log.sample_times()

    if recording_output_dir is not None:
        recording_output_dir.mkdir(parents=True, exist_ok=True)
        html = meshcat.StaticHtml()
        with open(
            recording_output_dir / f"{sdf_file_path.stem}.html", "w", encoding="utf-8"
        ) as f:
            f.write(html)

    return SimulationResult(
        simulation_time=end_time - start_time,
        states=states,
        sample_times=sample_times,
    )


def eval_static_equilibrium(
    visual_obj_mesh_path: Path,
    sdf_file_paths: list[Path],
    physics_props: dict[str, float],
    recording_output_dir: Path | None = None,
) -> EvalResult:
    """Evaluate the static equilibrium of a set of simulation assets.

    Args:
        visual_obj_mesh_path (Path): The path to the visual OBJ object mesh.
        sdf_file_paths (list[Path]): The paths to the SDF files to evaluate.
        physics_props (dict[str, float]): The physics properties of the object.
        recording_output_dir (Path | None): The directory to save the recording to.

    Returns:
        EvalResult: The evaluation result.
    """
    # Create the GT SDF file.
    gt_sdf_path = (
        visual_obj_mesh_path.parent / f"{visual_obj_mesh_path.stem}_rigid_hydro.sdf"
    )
    create_rigid_hydro_sdf_file(
        output_path=gt_sdf_path,
        visual_obj_mesh_path=visual_obj_mesh_path,
        mesh_for_physics_path=visual_obj_mesh_path,
        mass=physics_props["mass"],
        hunt_crossley_dissipation=physics_props.get("hunt_crossley_dissipation"),
        mu_dynamic=physics_props["mu_dynamic"],
        mu_static=physics_props["mu_static"],
    )
    all_sdf_file_paths = [gt_sdf_path] + sdf_file_paths

    meshcat: Meshcat = StartMeshcat()

    # Run the simulations.
    # TODO: Run in parallel if slow.
    sim_results: list[SimulationResult] = []
    for sdf_file_path in all_sdf_file_paths:
        sim_results.append(
            _run_single_static_equilibirum_sim(
                sdf_file_path, meshcat, recording_output_dir
            )
        )

    # Compute the metrics.
    gt_results = sim_results[0]
    processed_results = sim_results[1:]
    orientation_considered_final_errors: list[float] = []
    orientation_considered_average_errors: list[float] = []
    for processed_result in processed_results:
        gt_states = gt_results.states
        processed_states = processed_result.states
        orientation_considered_final_error = (
            orientation_considered_final_displacement_error(
                gt_state_trajectory=gt_states, state_trajectory=processed_states
            )
        )
        orientation_considered_average_error = (
            orientation_considered_average_displacement_error(
                gt_state_trajectory=gt_states, state_trajectory=processed_states
            )
        )
        orientation_considered_final_errors.append(orientation_considered_final_error)
        orientation_considered_average_errors.append(
            orientation_considered_average_error
        )

    return EvalResult(
        sdf_file_paths=sdf_file_paths,
        orientation_considered_final_errors=orientation_considered_final_errors,
        orientation_considered_average_errors=orientation_considered_average_errors,
        simulation_times=[r.simulation_time for r in processed_results],
    )
