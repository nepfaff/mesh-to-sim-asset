import json
import logging
import shutil
import traceback

from pathlib import Path

import numpy as np

from tqdm import tqdm

from mesh_to_sim_asset.canonicalize import canonicalize_gltf
from mesh_to_sim_asset.dynamic_eval import eval_static_equilibrium
from mesh_to_sim_asset.mesh_conversion import (
    convert_blend_to_gltf,
    convert_dae_to_gltf,
    convert_fbx_to_gltf,
    convert_glb_to_gltf,
    convert_glb_to_obj,
    convert_obj_to_gltf,
    convert_ply_to_gltf,
    convert_usd_to_gltf,
)
from mesh_to_sim_asset.mesh_simplification import (
    run_rolopoly,
    simplify_mesh_non_destructive,
)
from mesh_to_sim_asset.mesh_to_vtk import convert_mesh_to_vtk
from mesh_to_sim_asset.physics import (
    compute_hydroelastic_modulus,
    get_physics_properties,
)
from mesh_to_sim_asset.sdformat import (
    create_convex_decomposition_sdf_file,
    create_vtk_sdf_file,
)
from mesh_to_sim_asset.util import (
    CollisionGeomType,
    OpenAIModelType,
    comprehensive_error_handling,
    get_basename_without_extension,
    get_metadata_if_exists,
)

# Set up logger.
logger = logging.getLogger(__name__)


def should_skip_existing(mesh_path: str, output_dir: str) -> bool:
    """Check if we should skip processing this mesh because output already exists.

    Args:
        mesh_path (str): Path to the mesh file.
        output_dir (str): Directory to store output files.

    Returns:
        bool: True if should skip, False otherwise.
    """
    mesh_path = Path(mesh_path)
    mesh_name = get_basename_without_extension(mesh_path)
    mesh_parent_dir = Path(output_dir)
    mesh_dir = mesh_parent_dir / mesh_name

    return mesh_dir.exists()


def convert_mesh_to_gltf(mesh_path: Path, gltf_path: Path) -> bool:
    """Convert input mesh to GLTF format.

    Args:
        mesh_path: Path to the input mesh file.
        gltf_path: Path where the GLTF file should be saved.

    Returns:
        True if conversion was successful, False otherwise.
    """
    try:
        mesh_suffix_lower = mesh_path.suffix.lower()
        if mesh_suffix_lower == ".obj":
            convert_obj_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower == ".ply":
            convert_ply_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower == ".fbx":
            convert_fbx_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower == ".blend":
            convert_blend_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower == ".dae":
            convert_dae_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower in [".usd", ".usda", ".usdz"]:
            convert_usd_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower == ".gltf":
            # This conversion ensures that we end up with a gltf file with separate
            # texture files as desired by Drake.
            convert_glb_to_gltf(mesh_path, gltf_path)
        elif mesh_suffix_lower == ".glb":
            convert_glb_to_gltf(mesh_path, gltf_path)
        else:
            logger.warning(f"Unsupported mesh file: {mesh_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to convert {mesh_path} to GLTF: {e}")
        return False


def get_and_save_physics_properties(
    gltf_path: Path,
    mesh_path: Path,
    mesh_dir: Path,
    mesh_name: str,
    is_metric: bool,
    keep_images: bool,
    use_cpu_rendering: bool,
    model_type: OpenAIModelType,
) -> dict:
    """Get physics properties for the mesh and save to JSON file.

    Args:
        gltf_path: Path to the GLTF mesh file.
        mesh_path: Path to the original mesh file.
        mesh_dir: Directory where output files are stored.
        mesh_name: Name of the mesh.
        is_metric: Whether the mesh is already in metric units.
        keep_images: Whether to keep the images used for the LLM analysis.
        use_cpu_rendering: Whether to use CPU rendering.
        model_type: The OpenAI model to use.

    Returns:
        Dictionary containing physics properties.
    """
    images_path = mesh_dir / "mesh_renders" if keep_images else None
    metadata = get_metadata_if_exists(mesh_path=mesh_path)
    physics_props = get_physics_properties(
        gltf_path=gltf_path,
        is_metric_scale=is_metric,
        images_path=images_path,
        use_cpu_rendering=use_cpu_rendering,
        metadata=metadata,
        model=model_type.get_model_id(),
    )

    # Save the physics properties to a json file.
    physics_props_path = mesh_dir / f"{mesh_name}_properties.json"
    with open(physics_props_path, "w", encoding="utf-8") as f:
        json.dump(physics_props, f, indent=2)

    return physics_props


def create_collision_mesh_obj(
    gltf_path: Path, mesh_dir: Path, mesh_name: str
) -> tuple[Path, Path]:
    """Create OBJ version for collision mesh processing.

    Args:
        gltf_path: Path to the GLTF mesh file.
        mesh_dir: Directory where output files are stored.
        mesh_name: Name of the mesh.

    Returns:
        Tuple of (rolopoly_path, copy_path) for the OBJ files.
    """
    mesh_path_rolopoly = mesh_dir / f"{mesh_name}_rolopoly.obj"
    convert_glb_to_obj(gltf_path, mesh_path_rolopoly)

    # Create a copy of the obj file for later use.
    mesh_path_rolopoly_copy = mesh_dir / f"{mesh_name}.obj"
    shutil.copy(mesh_path_rolopoly, mesh_path_rolopoly_copy)

    return mesh_path_rolopoly, mesh_path_rolopoly_copy


def create_coacd_collision_geometry(
    mesh_dir: Path,
    mesh_name: str,
    gltf_path: Path,
    mesh_path_rolopoly_copy: Path,
    mesh_path_rolopoly: Path,
    physics_props: dict,
    multiple_collision_geoms: bool,
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.001,
) -> Path:
    """Create CoACD collision geometry and SDF file.

    Args:
        mesh_dir: Directory where output files are stored.
        mesh_name: Name of the mesh.
        gltf_path: Path to the GLTF visual mesh.
        mesh_path_rolopoly_copy: Path to the physics mesh copy.
        mesh_path_rolopoly: Path to the collision mesh.
        physics_props: Physics properties for the mesh.
        multiple_collision_geoms: Whether multiple collision geometry types are being
            created.
        resolution_values: List of resolution values to try for CoACD iterative
            refinement. If None, uses single-pass behavior.
        volume_change_threshold: Maximum absolute volume improvement (m³) per additional
            part to continue refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
            Default is 0.001 m³.

    Returns:
        Path to the created SDF file.
    """
    sdf_path = (
        mesh_dir / f"{mesh_name}_coacd.sdf"
        if multiple_collision_geoms
        else mesh_dir / f"{mesh_name}.sdf"
    )
    logger.info(f"Creating CoACD SDF file for {mesh_name} at {sdf_path}")
    create_convex_decomposition_sdf_file(
        output_path=sdf_path,
        mesh_parts_dir_name=f"{mesh_name}_coacd",
        visual_mesh_path=gltf_path,
        mesh_for_physics_path=mesh_path_rolopoly_copy,
        collision_mesh_path=mesh_path_rolopoly,
        mass=physics_props["mass"],
        material_modulus=physics_props["material_modulus"],
        hunt_crossley_dissipation=physics_props.get("hunt_crossley_dissipation"),
        mu_dynamic=physics_props["mu_dynamic"],
        mu_static=physics_props["mu_static"],
        use_coacd=True,
        resolution_values=resolution_values,
        volume_change_threshold=volume_change_threshold,
    )
    logger.info(f"CoACD SDF file created for {mesh_name} at {sdf_path}")
    return sdf_path


def create_vhacd_collision_geometry(
    mesh_dir: Path,
    mesh_name: str,
    gltf_path: Path,
    mesh_path_rolopoly_copy: Path,
    mesh_path_rolopoly: Path,
    physics_props: dict,
    multiple_collision_geoms: bool,
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.001,
) -> Path:
    """Create VHACD collision geometry and SDF file.

    Args:
        mesh_dir: Directory where output files are stored.
        mesh_name: Name of the mesh.
        gltf_path: Path to the GLTF visual mesh.
        mesh_path_rolopoly_copy: Path to the physics mesh copy.
        mesh_path_rolopoly: Path to the collision mesh.
        physics_props: Physics properties for the mesh.
        multiple_collision_geoms: Whether multiple collision geometry types are being
            created.
        resolution_values: List of resolution values to try for CoACD iterative
            refinement. If None, uses single-pass behavior.
        volume_change_threshold: Maximum absolute volume improvement (m³) per additional
            part to continue refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
            Default is 0.01 m³.

    Returns:
        Path to the created SDF file.
    """
    sdf_path = (
        mesh_dir / f"{mesh_name}_vhacd.sdf"
        if multiple_collision_geoms
        else mesh_dir / f"{mesh_name}.sdf"
    )
    logger.info(f"Creating VHACD SDF file for {mesh_name} at {sdf_path}")
    create_convex_decomposition_sdf_file(
        output_path=sdf_path,
        mesh_parts_dir_name=f"{mesh_name}_vhacd",
        visual_mesh_path=gltf_path,
        mesh_for_physics_path=mesh_path_rolopoly_copy,
        collision_mesh_path=mesh_path_rolopoly,
        mass=physics_props["mass"],
        material_modulus=physics_props["material_modulus"],
        hunt_crossley_dissipation=physics_props.get("hunt_crossley_dissipation"),
        mu_dynamic=physics_props["mu_dynamic"],
        mu_static=physics_props["mu_static"],
        use_coacd=False,
        resolution_values=resolution_values,
        volume_change_threshold=volume_change_threshold,
    )
    logger.info(f"VHACD SDF file created for {mesh_name} at {sdf_path}")
    return sdf_path


def create_vtk_collision_geometry(
    mesh_dir: Path,
    mesh_name: str,
    gltf_path: Path,
    mesh_path_rolopoly_copy: Path,
    mesh_path_rolopoly: Path,
    physics_props: dict,
    rolopoly_timeout: int,
    multiple_collision_geoms: bool,
) -> Path | None:
    """Create VTK collision geometry and SDF file using RoLoPoly simplification.

    Args:
        mesh_dir: Directory where output files are stored.
        mesh_name: Name of the mesh.
        gltf_path: Path to the GLTF visual mesh.
        mesh_path_rolopoly_copy: Path to the physics mesh copy.
        mesh_path_rolopoly: Path to the collision mesh.
        physics_props: Physics properties for the mesh.
        rolopoly_timeout: The RoLoPoly timeout in seconds.
        multiple_collision_geoms: Whether multiple collision geometry types are being
            created.

    Returns:
        Path to the created SDF file, or None if creation failed.
    """
    # Use a finer simplification for big meshes with thin parts (e.g., a
    # table with thin legs).
    is_big_with_thin_parts = physics_props["big_with_thin_parts"]
    screen_size = 250 if is_big_with_thin_parts else 100

    # Clean up the mesh non-destructively before RoLoPoly simplification.
    try:
        logger.info(f"Cleaning up mesh {mesh_name} before RoLoPoly")
        simplify_mesh_non_destructive(mesh_path_rolopoly, mesh_path_rolopoly)
        logger.info(f"Mesh cleanup completed for {mesh_name}")
    except Exception as e:
        logger.warning(
            f"Mesh cleanup failed for {mesh_name}: {e}. Proceeding with original mesh."
        )

    # Sequentially simplify with RoLoPoly until Tetgen succeeds.
    vtk_path = None
    for i in tqdm(
        range(5),
        desc=f"  Simplifying with RoLoPoly ({mesh_name})",
        leave=False,
        position=1,
    ):
        try:
            # Simplify with RoLoPoly.
            output_folder = mesh_dir / f"{mesh_name}_rolopoly"
            logger.info(f"Running RoLoPoly for {mesh_name} at {output_folder}")
            run_rolopoly(
                input_path=mesh_path_rolopoly,
                output_folder=output_folder,
                timeout=rolopoly_timeout,
                screen_size=screen_size,
            )
            logger.info(f"RoLoPoly completed for {mesh_name} at {output_folder}")

            # Replace mesh with the simplified version and clean up.
            new_mesh_path = output_folder / f"{mesh_name}_rolopoly_ours_final.obj"
            shutil.copy(new_mesh_path, mesh_path_rolopoly)
            shutil.rmtree(output_folder)

            logger.info(f"Converting {mesh_name} to vtk")
            vtk_path = convert_mesh_to_vtk(input_path=mesh_path_rolopoly)
            logger.info(f"VTK file created for {mesh_name} at {vtk_path}")
            break
        except Exception as e:
            logger.error(
                f"Failed to convert {mesh_name} to vtk on iteration {i+1}: {e}"
            )

            if "RoLoPoly timed out" in str(e):
                # No point to try again if failed due to timeout.
                break
            continue

    if vtk_path is not None:
        # Compute hydroelastic modulus.
        hydroelastic_modulus = compute_hydroelastic_modulus(
            material_modulus=physics_props["material_modulus"],
            vtk_file_path=vtk_path,
        )

        if is_big_with_thin_parts:
            # Scale the modulus to prevent penetration at the thin parts.
            hydroelastic_modulus *= 1e2

        # Create the SDFormat file.
        sdf_path = (
            mesh_dir / f"{mesh_name}_vtk.sdf"
            if multiple_collision_geoms
            else mesh_dir / f"{mesh_name}.sdf"
        )
        logger.info(f"Creating VTK SDF file for {mesh_name} at {sdf_path}")
        create_vtk_sdf_file(
            output_path=sdf_path,
            visual_mesh_path=gltf_path,
            mesh_for_physics_path=mesh_path_rolopoly_copy,
            collision_mesh_path=vtk_path,
            mass=physics_props["mass"],
            hydroelastic_modulus=hydroelastic_modulus,
            hunt_crossley_dissipation=physics_props.get("hunt_crossley_dissipation"),
            mu_dynamic=physics_props["mu_dynamic"],
            mu_static=physics_props["mu_static"],
        )
        logger.info(f"VTK SDF file created for {mesh_name} at {sdf_path}")
        return sdf_path
    else:
        logger.warning(f"Failed to convert {mesh_name} to vtk after 5 iterations.")
        return None


def evaluate_and_select_best_collision_geometry(
    sdf_paths: list[Path],
    mesh_path_rolopoly_copy: Path,
    physics_props: dict,
    mesh_dir: Path,
    mesh_name: str,
    debug: bool,
) -> None:
    """Evaluate simulation assets and select the best collision geometry.

    Args:
        sdf_paths: List of paths to SDF files to evaluate.
        mesh_path_rolopoly_copy: Path to the physics mesh copy.
        physics_props: Physics properties for the mesh.
        mesh_dir: Directory where output files are stored.
        mesh_name: Name of the mesh.
        debug: Whether to save debug information.
    """
    if len(sdf_paths) <= 1:
        return

    # Evaluate simulation assets.
    eval_result = eval_static_equilibrium(
        visual_obj_mesh_path=mesh_path_rolopoly_copy,
        sdf_file_paths=sdf_paths,
        physics_props=physics_props,
        recording_output_dir=(mesh_dir / "sim_eval_recordings" if debug else None),
    )
    if debug:
        eval_result.save_to_disk(mesh_dir / "eval_result.json")

    # Choose the best SDF file based on the eval result, giving a slight
    # preference to the VTK SDF file.
    sorted_indices = np.argsort(eval_result.orientation_considered_average_errors)
    sdf_paths_sorted: list[Path] = [sdf_paths[i] for i in sorted_indices]
    errors_sorted: list[float] = [
        eval_result.orientation_considered_average_errors[i] for i in sorted_indices
    ]
    vtk_idx = [
        i for i, sdf_path in enumerate(sdf_paths) if sdf_path.name.endswith("_vtk.sdf")
    ]
    best_sdf_path = None
    if len(vtk_idx) > 0:
        # Choose the best if it is better than the VTK SDF file by a
        # factor of 2.5.
        best_score = errors_sorted[0]
        vtk_score = eval_result.orientation_considered_average_errors[vtk_idx[0]]
        if best_score * 2.5 < vtk_score and abs(best_score - vtk_score) > 0.01:
            best_sdf_path = sdf_paths_sorted[0]
        else:
            best_sdf_path = sdf_paths[vtk_idx[0]]
        logger.info(
            f"Using {best_sdf_path} as the best SDF file. Best score: {best_score}, "
            f"VTK score: {vtk_score}, Abs diff: {abs(best_score - vtk_score)}"
        )
    else:
        best_sdf_path = sdf_paths_sorted[0]

    # Create a copy of the best SDF file.
    best_sdf_path_copy = mesh_dir / f"{mesh_name}.sdf"
    shutil.copy(best_sdf_path, best_sdf_path_copy)


def process_single_mesh(
    mesh_path: Path,
    mesh_dir: Path,
    is_metric: bool,
    canonicalize: bool,
    keep_images: bool,
    use_cpu_rendering: bool,
    collision_geom_type: list[CollisionGeomType],
    rolopoly_timeout: int,
    model_type: OpenAIModelType,
    debug: bool,
    only_process_single_objects: bool,
    only_process_textured: bool,
    only_process_simulatable: bool,
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.001,
) -> None:
    """Process a single mesh file into simulation-ready asset.

    Args:
        mesh_path: Path to the mesh file.
        mesh_dir: Directory to store output files.
        is_metric: Whether the mesh is already in metric units.
        canonicalize: Whether to canonicalize the mesh.
        keep_images: Whether to keep the images used for the LLM analysis.
        use_cpu_rendering: Whether to use CPU rendering.
        collision_geom_type: The type(s) of collision geometry to use for the mesh.
        rolopoly_timeout: The RoLoPoly timeout in seconds.
        model_type: The OpenAI model to use.
        debug: Whether to log debug information.
        only_process_single_objects: Whether to skip assets that are not identified
            as single objects by the VLM analysis.
        only_process_textured: Whether to skip assets that are not identified
            as textured by the VLM analysis.
        only_process_simulatable: Whether to skip assets that are not identified
            as simulatable by the VLM analysis.
        resolution_values: List of resolution values to try for CoACD iterative
            refinement. If None, uses single-pass behavior.
        volume_change_threshold: Maximum absolute volume improvement (m³) per additional
            part to continue refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
            Default is 0.001 m³.

    Raises:
        Exception: If processing fails at any step.
    """
    mesh_name = get_basename_without_extension(mesh_path)
    logger.info(f"Processing {mesh_name} at {mesh_path}")

    # Convert the input mesh to a GLTF mesh.
    gltf_path = mesh_dir / f"{mesh_name}.gltf"
    if not convert_mesh_to_gltf(mesh_path, gltf_path):
        return

    # Get physics properties including mesh pose.
    physics_props = get_and_save_physics_properties(
        gltf_path=gltf_path,
        mesh_path=mesh_path,
        mesh_dir=mesh_dir,
        mesh_name=mesh_name,
        is_metric=is_metric,
        keep_images=keep_images,
        use_cpu_rendering=use_cpu_rendering,
        model_type=model_type,
    )

    # Skip processing if not a single object when the flag is enabled.
    if only_process_single_objects and not physics_props.get("is_single_object", True):
        logger.warning(
            f"Skipping {mesh_name} because it is not identified as a single object "
            "by the VLM analysis."
        )
        return

    # Skip processing if not textured when the flag is enabled.
    if only_process_textured and not physics_props.get("is_textured", True):
        logger.warning(
            f"Skipping {mesh_name} because it is not identified as textured "
            "by the VLM analysis."
        )
        return

    # Skip processing if not simulatable when the flag is enabled.
    if only_process_simulatable and not physics_props.get("is_simulatable", True):
        logger.warning(
            f"Skipping {mesh_name} because it is not identified as simulatable "
            "by the VLM analysis."
        )
        return

    # Canonicalize the mesh if requested.
    if canonicalize:
        canonicalize_gltf(
            input_gltf_path=gltf_path,
            output_gltf_path=gltf_path,
            canonical_orientation=physics_props["canonical_orientation"],
            scale=physics_props["scale"],
            placement_options=physics_props["placement_options"],
        )

    # Create OBJ version for collision mesh processing.
    mesh_path_rolopoly, mesh_path_rolopoly_copy = create_collision_mesh_obj(
        gltf_path=gltf_path, mesh_dir=mesh_dir, mesh_name=mesh_name
    )

    # Create collision geometries based on specified types.
    multiple_collision_geoms = len(collision_geom_type) > 1
    sdf_paths: list[Path] = []

    if CollisionGeomType.CoACD in collision_geom_type:
        sdf_path = create_coacd_collision_geometry(
            mesh_dir=mesh_dir,
            mesh_name=mesh_name,
            gltf_path=gltf_path,
            mesh_path_rolopoly_copy=mesh_path_rolopoly_copy,
            mesh_path_rolopoly=mesh_path_rolopoly,
            physics_props=physics_props,
            multiple_collision_geoms=multiple_collision_geoms,
            resolution_values=resolution_values,
            volume_change_threshold=volume_change_threshold,
        )
        sdf_paths.append(sdf_path)

    if CollisionGeomType.VHACD in collision_geom_type:
        sdf_path = create_vhacd_collision_geometry(
            mesh_dir=mesh_dir,
            mesh_name=mesh_name,
            gltf_path=gltf_path,
            mesh_path_rolopoly_copy=mesh_path_rolopoly_copy,
            mesh_path_rolopoly=mesh_path_rolopoly,
            physics_props=physics_props,
            multiple_collision_geoms=multiple_collision_geoms,
            resolution_values=resolution_values,
            volume_change_threshold=volume_change_threshold,
        )
        sdf_paths.append(sdf_path)

    if CollisionGeomType.VTK in collision_geom_type:
        sdf_path = create_vtk_collision_geometry(
            mesh_dir=mesh_dir,
            mesh_name=mesh_name,
            gltf_path=gltf_path,
            mesh_path_rolopoly_copy=mesh_path_rolopoly_copy,
            mesh_path_rolopoly=mesh_path_rolopoly,
            physics_props=physics_props,
            rolopoly_timeout=rolopoly_timeout,
            multiple_collision_geoms=multiple_collision_geoms,
        )
        if sdf_path is not None:
            sdf_paths.append(sdf_path)

    if len(sdf_paths) == 0:
        logger.warning(f"Failed to create any SDF files for {mesh_name}.")
        return

    # Evaluate and select best collision geometry if multiple types were created.
    evaluate_and_select_best_collision_geometry(
        sdf_paths=sdf_paths,
        mesh_path_rolopoly_copy=mesh_path_rolopoly_copy,
        physics_props=physics_props,
        mesh_dir=mesh_dir,
        mesh_name=mesh_name,
        debug=debug,
    )

    # Clean up temporary files.
    mesh_path_rolopoly.unlink(missing_ok=True)
    mesh_path_rolopoly.with_suffix(".mtl").unlink(missing_ok=True)
    mesh_path_rolopoly_copy.unlink(missing_ok=True)


def process_meshes(
    mesh_paths: list[str],
    output_dir: str,
    is_metric: bool,
    canonicalize: bool,
    keep_images: bool,
    use_cpu_rendering: bool,
    collision_geom_type: list[CollisionGeomType],
    rolopoly_timeout: int,
    model_type: OpenAIModelType,
    debug: bool,
    only_process_single_objects: bool,
    only_process_textured: bool,
    only_process_simulatable: bool,
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.001,
) -> None:
    """Process a list of mesh files into simulation-ready assets.

    Args:
        mesh_paths: List of paths to mesh files to process.
        output_dir: Directory to store output files.
        is_metric: Whether the mesh is already in metric units.
        canonicalize: Whether to canonicalize the mesh.
        keep_images: Whether to keep the images used for the LLM analysis.
        use_cpu_rendering: Whether to use CPU rendering.
        collision_geom_type: The type(s) of collision geometry
            to use for the mesh. Can specify multiple types.
        rolopoly_timeout: The RoLoPoly timeout in seconds.
        model_type: The OpenAI model to use.
        debug: Whether to log debug information.
        only_process_single_objects: Whether to skip assets that are not identified
            as single objects by the VLM analysis.
        only_process_textured: Whether to skip assets that are not identified
            as textured by the VLM analysis.
        only_process_simulatable: Whether to skip assets that are not identified
            as simulatable by the VLM analysis.
        resolution_values: List of resolution values to try for CoACD iterative
            refinement. If None, uses single-pass behavior.
        volume_change_threshold: Maximum absolute volume improvement per additional
            part to continue refinement. Measures
            -(V_n - V_{n-1}) / (N_n - N_{n-1}). Default is 0.001 m³.
    """
    # Folder to store the mesh files in.
    mesh_parent_dir = Path(output_dir)
    mesh_parent_dir.mkdir(parents=True, exist_ok=True)

    for mesh_path in tqdm(mesh_paths, desc="Processing meshes"):
        try:
            with comprehensive_error_handling():
                mesh_path = Path(mesh_path)
                mesh_name = get_basename_without_extension(mesh_path)
                mesh_dir = mesh_parent_dir / mesh_name

                logger.info(f"\n\n\nProcessing {mesh_name} at {mesh_path}")

                # Create a folder for the output meshes.
                mesh_dir.mkdir(parents=True, exist_ok=True)

                # Process the single mesh.
                process_single_mesh(
                    mesh_path=mesh_path,
                    mesh_dir=mesh_dir,
                    is_metric=is_metric,
                    canonicalize=canonicalize,
                    keep_images=keep_images,
                    use_cpu_rendering=use_cpu_rendering,
                    collision_geom_type=collision_geom_type,
                    rolopoly_timeout=rolopoly_timeout,
                    model_type=model_type,
                    debug=debug,
                    only_process_single_objects=only_process_single_objects,
                    only_process_textured=only_process_textured,
                    only_process_simulatable=only_process_simulatable,
                    resolution_values=resolution_values,
                    volume_change_threshold=volume_change_threshold,
                )

                logger.info(
                    f"Successfully processed {mesh_name}. Output files: {mesh_dir}"
                )

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user (Ctrl+C)")
            raise  # Re-raise to allow proper termination
        except Exception as e:
            mesh_name = get_basename_without_extension(Path(mesh_path))
            logger.error(f"Failed to process {mesh_name}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            continue
