"""
Process 3D meshes into simulation-ready assets with physics properties.

Input is a folder structure with ".obj", ".ply", ".fbx", ".blend", ".dae", ".usd",
".usda", ".usdz", ".glb", or ".gltf" files. All mesh files should have unique names
(ideally names have some meaning but this isn't necessary).
Optionally, metadata can be provided in json format at "{stem}_metadata.json".

Note that all input meshes are assumed to have unique file names.

This script:
1. Simplifies meshes using RoLoPoly for better physics simulation
2. Converts meshes to GLTF format with correct coordinate system
3. Estimates physics properties using LLM analysis:
   - Mass and material properties
   - Canonical orientation and dimensions
   - Friction coefficients
4. Creates SDFormat files with:
   - Visual and collision meshes
   - Physics properties
   - Pose that places the mesh in its canonical orientation with bottom at z=0

The resulting assets can be used in physics simulation environments like Drake.
"""

import argparse
import logging

from pathlib import Path

from mesh_to_sim_asset.create_drake_assets_from_geometry import (
    process_meshes,
    should_skip_existing,
)
from mesh_to_sim_asset.util import (
    CollisionGeomType,
    OpenAIModelType,
    find_mesh_files,
    get_basename_without_extension,
)

# Set up logger.
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process 3D meshes into simulation-ready assets with physics "
        "properties."
    )
    parser.add_argument(
        "input_path",
        help="Path to a mesh file or directory containing mesh files to process.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to store output files.",
    )
    parser.add_argument(
        "--is-metric",
        "-m",
        action="store_true",
        help="Whether the mesh is already in metric units. This will prevent rescaling.",
    )
    parser.add_argument(
        "--canonicalize",
        "-c",
        action="store_true",
        help="Whether to canonicalize the mesh pose.",
    )
    parser.add_argument(
        "--keep-images",
        "-k",
        action="store_true",
        help="Whether to keep the images used for the LLM analysis.",
    )
    parser.add_argument(
        "--use-cpu-rendering",
        "-u",
        action="store_true",
        help="Whether to use CPU rendering.",
    )
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Whether to skip meshes that already have an entry in the output directory.",
    )
    parser.add_argument(
        "--collision-geom-type",
        "-g",
        type=CollisionGeomType,
        default=[CollisionGeomType.CoACD, CollisionGeomType.VTK],
        nargs="+",
        help="The type(s) of collision geometry to use for the mesh. Can specify "
        "multiple types.",
    )
    parser.add_argument(
        "--rolopoly_timeout",
        type=int,
        default=3600,
        help="The RoLoPoly timeout in seconds. Useful to increase on weaker CPU "
        "machines or for huge input meshes.",
    )
    parser.add_argument(
        "--model",
        "-z",
        type=OpenAIModelType,
        default=OpenAIModelType.O3,
        help="Recommended to use O3 for best performance and GPT4 for lower cost. "
        "O3 is especially useful for spatial reasoning for determining the up axis "
        "and likely overkill if the assets are already in canonical orientations with "
        "z up.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether to log debug information.",
    )
    parser.add_argument(
        "--only-process-single-objects",
        action="store_true",
        help="Whether to skip all assets that are not identified as single objects "
        "by the VLM analysis.",
    )
    parser.add_argument(
        "--only-process-textured",
        action="store_true",
        help="Whether to skip all assets that are not identified as textured "
        "by the VLM analysis.",
    )
    parser.add_argument(
        "--only-process-simulatable",
        action="store_true",
        help="Whether to skip all assets that are not identified as simulatable "
        "by the VLM analysis.",
    )
    parser.add_argument(
        "--resolution-values",
        "-r",
        type=float,
        nargs="*",
        default=None,
        help="List of resolution values to try for CoACD iterative refinement. "
        "These are threshold values where lower values = higher resolution. "
        "Example: --resolution-values 0.1 0.05 0.01. If not specified, uses "
        "single-pass behavior with a value of 0.05.",
    )
    parser.add_argument(
        "--volume-change-threshold",
        "-v",
        type=float,
        default=0.00003,
        help="Maximum absolute volume improvement (m³) per additional part to continue "
        "refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}) where V_n and N_n "
        "are volume and part count at iteration n. Default is 0.00003 m³. Only used when "
        "--resolution-values is specified. This is the absolute volume improvement per "
        "additional part to continue refinement.",
    )
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    input_path = Path(args.input_path)
    if input_path.is_file():
        mesh_paths = [str(input_path)]
    else:
        mesh_paths = [str(p) for p in find_mesh_files(str(input_path))]

    if not mesh_paths:
        logger.warning(f"No mesh files found in {input_path}")
        return

    # Filter out meshes that should be skipped.
    if args.skip_existing:
        meshes_to_process = []
        for mesh_path in mesh_paths:
            if should_skip_existing(mesh_path, args.output_dir):
                mesh_name = get_basename_without_extension(Path(mesh_path))
                logger.info(
                    f"Skipping {mesh_name} because it already has an entry in the "
                    f"{args.output_dir} directory."
                )
            else:
                meshes_to_process.append(mesh_path)
        mesh_paths = meshes_to_process

    if not mesh_paths:
        logger.info("No meshes to process after filtering.")
        return

    process_meshes(
        mesh_paths=mesh_paths,
        output_dir=args.output_dir,
        is_metric=args.is_metric,
        canonicalize=args.canonicalize,
        keep_images=args.keep_images,
        use_cpu_rendering=args.use_cpu_rendering,
        collision_geom_type=args.collision_geom_type,
        rolopoly_timeout=args.rolopoly_timeout,
        model_type=args.model,
        debug=args.debug,
        only_process_single_objects=args.only_process_single_objects,
        only_process_textured=args.only_process_textured,
        only_process_simulatable=args.only_process_simulatable,
        resolution_values=args.resolution_values,
        volume_change_threshold=args.volume_change_threshold,
    )


if __name__ == "__main__":
    args = parse_args()

    # Create output directory first to ensure it exists for logging.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to write to a file in the output directory.
    log_level = logging.INFO if args.debug else logging.WARNING
    log_file = output_dir / "create_drake_asset_from_geometry.log"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",  # Overwrite log file on each run
    )

    main(args)
