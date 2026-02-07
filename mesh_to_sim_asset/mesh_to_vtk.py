import argparse
import logging
import os
import signal
import sys

from contextlib import contextmanager
from pathlib import Path

import tetgen

from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import MeshSource, RefineVolumeMeshIntoVtkFileContents

from mesh_to_sim_asset.mesh_conversion import (
    convert_blend_to_gltf,
    convert_dae_to_gltf,
    convert_fbx_to_gltf,
    convert_glb_to_obj,
    convert_usd_to_gltf,
)
from mesh_to_sim_asset.sdformat import create_vtk_sdf_file

# Set up logger.
logger = logging.getLogger(__name__)


@contextmanager
def handle_segfault():
    """Context manager to handle segmentation faults gracefully.

    Yields:
        None: The context manager yields control to the wrapped code.
    """

    def segfault_handler(signum, frame):
        raise RuntimeError("Segmentation fault occurred during mesh processing.")

    # Set up the signal handler.
    original_handler = signal.signal(signal.SIGSEGV, segfault_handler)
    try:
        yield
    finally:
        # Restore the original signal handler.
        signal.signal(signal.SIGSEGV, original_handler)


def convert_mesh_to_vtk(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert a mesh file to VTK format.

    Args:
        input_path (str): Path to the input mesh (GLTF, OBJ, PLY, FBX, Blender, DAE, or USD).
        output_path (str, optional): Path for the output VTK file. If not provided,
            will use input filename with .vtk extension.

    Returns:
        str: Path to the generated VTK file.

    Raises:
        ValueError: If the input file format is not supported.
        RuntimeError: If the conversion process fails or encounters a segmentation fault.
    """
    # Convert to string paths.
    input_path = input_path.as_posix()
    if output_path is not None:
        output_path = output_path.as_posix()

    # Determine output path.
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".vtk"

    # Get file extension.
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    try:
        # Convert GLTF to OBJ if needed.
        if ext == ".gltf":
            intermediate_path = Path(os.path.splitext(input_path)[0] + ".obj")
            convert_glb_to_obj(
                input_path=input_path,
                output_path=intermediate_path,
            )
        elif ext in [".obj", ".ply"]:
            intermediate_path = input_path
        elif ext == ".fbx":
            # Convert FBX to GLTF first, then to OBJ
            gltf_intermediate_path = Path(os.path.splitext(input_path)[0] + ".gltf")
            convert_fbx_to_gltf(
                input_path=Path(input_path),
                output_path=gltf_intermediate_path,
            )
            intermediate_path = Path(os.path.splitext(input_path)[0] + ".obj")
            convert_glb_to_obj(
                input_path=gltf_intermediate_path,
                output_path=intermediate_path,
            )
        elif ext == ".blend":
            # Convert Blender file to GLTF first, then to OBJ
            gltf_intermediate_path = Path(os.path.splitext(input_path)[0] + ".gltf")
            convert_blend_to_gltf(
                input_path=Path(input_path),
                output_path=gltf_intermediate_path,
            )
            intermediate_path = Path(os.path.splitext(input_path)[0] + ".obj")
            convert_glb_to_obj(
                input_path=gltf_intermediate_path,
                output_path=intermediate_path,
            )
        elif ext == ".dae":
            # Convert DAE to GLTF first, then to OBJ
            gltf_intermediate_path = Path(os.path.splitext(input_path)[0] + ".gltf")
            convert_dae_to_gltf(
                input_path=Path(input_path),
                output_path=gltf_intermediate_path,
            )
            intermediate_path = Path(os.path.splitext(input_path)[0] + ".obj")
            convert_glb_to_obj(
                input_path=gltf_intermediate_path,
                output_path=intermediate_path,
            )
        elif ext in [".usd", ".usda", ".usdz"]:
            # Convert USD to GLTF first, then to OBJ
            gltf_intermediate_path = Path(os.path.splitext(input_path)[0] + ".gltf")
            convert_usd_to_gltf(
                input_path=Path(input_path),
                output_path=gltf_intermediate_path,
            )
            intermediate_path = Path(os.path.splitext(input_path)[0] + ".obj")
            convert_glb_to_obj(
                input_path=gltf_intermediate_path,
                output_path=intermediate_path,
            )
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Supported formats are: .gltf, .obj, "
                ".ply, .fbx, .blend, .dae, .usd, .usda, .usdz"
            )

        # Convert to vtk with segmentation fault handling.
        with handle_segfault():
            tgen = tetgen.TetGen(intermediate_path)
            tgen.tetrahedralize(
                quality=True,
                nobisect=True,
                nomergefacet=True,
                nomergevertex=True,
                vtksurfview=True,
                vtkview=True,
                verbose=False,
            )
            tgen.write(output_path, binary=False)
    except Exception as e:
        raise RuntimeError(f"Failed to convert mesh to VTK: {str(e)}")

    # Run mesh refinement.
    output_path = Path(output_path)
    refined_vtk_str = RefineVolumeMeshIntoVtkFileContents(MeshSource(output_path))
    output_path.write_text(refined_vtk_str)

    # Check if can import into Drake. This requires creating a temporary .sdf file.
    tmp_sdf_file_path = str(output_path)[:-4] + "_tmp.sdf"
    create_vtk_sdf_file(
        output_path=Path(tmp_sdf_file_path),
        visual_mesh_path=Path(input_path),
        collision_mesh_path=output_path,
        mass=1.0,
        hydroelastic_modulus=1e6,
        mesh_for_physics_path=input_path,
    )
    # Convert to absolute path for proper URI resolution.
    tmp_sdf_file_path_abs = Path(tmp_sdf_file_path).resolve().as_posix()
    scenario_data = f"""
        directives:
        - add_model:
            name: model
            file: file://{tmp_sdf_file_path_abs}
    """
    scenario = LoadScenario(data=scenario_data)
    try:
        MakeHardwareStation(scenario=scenario)
    except Exception as e:
        raise RuntimeError(f"Failed to import mesh into Drake: {str(e)}")
    finally:
        # Clean up temporary file.
        os.remove(tmp_sdf_file_path)

    return Path(output_path)


def main(args: argparse.Namespace):
    """Command-line interface for mesh to VTK conversion."""
    try:
        convert_mesh_to_vtk(args.mesh_path, args.output)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh_path",
        type=str,
        help="The path to the input mesh (GLTF, OBJ, or PLY).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="The path for the output VTK file. If not provided, will use input "
        "filename with .vtk extension.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    # Configure logging.
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    main(args)
