"""
Make simulation asset formats (USD, URDF, SDF, MJX) Drake-compatible. Works for
articulated objects. USD textures/ materials are preserved through GLTF export.

Note that all input files are assumed to have unique file names.
"""

import argparse
import logging
import traceback

from pathlib import Path

from tqdm import tqdm

from mesh_to_sim_asset.make_asset_drake_compatible import (
    convert_asset_to_drake_sdf,
    copy_generated_outputs_to_target_dir,
    make_sdf_geometries_simulatable,
    should_skip_existing,
)
from mesh_to_sim_asset.util import (
    CollisionGeomType,
    OpenAIModelType,
    comprehensive_error_handling,
    get_basename_without_extension,
)

# Set up logger.
logger = logging.getLogger(__name__)


def find_asset_files(input_path: str) -> list[Path]:
    """Find asset files (USD, URDF, SDF) in the given path.

    Args:
        input_path: Path to search for asset files.

    Returns:
        List of asset file paths.
    """
    asset_extensions = [".usd", ".usda", ".usdc", ".urdf", ".sdf", ".xml"]
    asset_files = []

    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in asset_extensions:
            asset_files.append(input_path)
    else:
        for ext in asset_extensions:
            # Search for both lowercase and uppercase extensions to be case-agnostic.
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                found_files = list(input_path.rglob(pattern))
                for file_path in found_files:
                    # For USD files, exclude files inside "resource" dirs.
                    if ext in [".usd", ".usda", ".usdc"]:
                        if "resource" in file_path.parts:
                            continue
                    # Avoid duplicates by checking if file is already in the list.
                    if file_path not in asset_files:
                        asset_files.append(file_path)

    return asset_files


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
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
        help="Whether to skip meshes that already have an entry in the output "
        "directory.",
    )
    parser.add_argument(
        "--collision-geom-type",
        "-g",
        type=CollisionGeomType,
        default=[CollisionGeomType.CoACD],
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
        "and likely overkill if the assets are already in canonical orientations "
        "with z up.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether to log debug information.",
    )
    parser.add_argument(
        "--resolution-values",
        "-r",
        type=float,
        nargs="*",
        default=None,
        help="List of resolution values to try for CoACD iterative refinement. "
        "These are threshold values where lower values = higher resolution. "
        "Example: --resolution-values 0.07 0.05 0.03. If not specified, uses "
        "single-pass behavior.",
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
    parser.add_argument(
        "--canonicalize",
        "-c",
        action="store_true",
        help="Canonicalize asset orientation: rotate so up=+Z, front=+Y, center XY, "
        "bottom at Z=0. This modifies all mesh files and updates SDF transforms.",
    )
    parser.add_argument(
        "--hydroelastic",
        action="store_true",
        help="Use hydroelastic contact instead of point contact. Adds hydroelastic "
        "proximity properties to collision geometries.",
    )
    parser.add_argument(
        "--max-convex-hull",
        "-m",
        type=int,
        default=-1,
        help="Maximum number of convex hulls for CoACD decomposition. "
        "-1 means no limit (default). Only works when merge is enabled.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function for processing articulated assets."""
    # Find asset files to process.
    input_path = Path(args.input_path)
    if input_path.is_file():
        asset_paths = [input_path]
    else:
        asset_paths = find_asset_files(str(input_path))

    if not asset_paths:
        logger.warning(f"No asset files found in {input_path}")
        return

    # Create output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out assets that should be skipped.
    if args.skip_existing:
        assets_to_process = []
        for asset_path in asset_paths:
            if should_skip_existing(asset_path, output_dir, args.collision_geom_type):
                asset_name = get_basename_without_extension(asset_path)
                logger.info(
                    f"Skipping {asset_name} because it already has an entry in the "
                    f"{output_dir} directory."
                )
            else:
                assets_to_process.append(asset_path)
        asset_paths = assets_to_process

    if not asset_paths:
        logger.info("No assets to process after filtering.")
        return

    # Process each asset.
    for asset_path in tqdm(asset_paths, desc="Processing assets"):
        try:
            with comprehensive_error_handling():
                asset_name = get_basename_without_extension(asset_path)
                asset_output_dir = output_dir / asset_name
                asset_output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"\n\n\nProcessing {asset_name} at {asset_path}")

                # Process in the original directory to preserve all references.
                original_dir = asset_path.parent

                # Convert asset to SDF format in original directory.
                sdf_path = convert_asset_to_drake_sdf(asset_path=asset_path)

                # Make geometries simulation-ready in original directory.
                simulatable_sdf_paths = make_sdf_geometries_simulatable(
                    sdf_path=sdf_path,
                    keep_images=args.keep_images,
                    use_cpu_rendering=args.use_cpu_rendering,
                    collision_geom_type=args.collision_geom_type,
                    rolopoly_timeout=args.rolopoly_timeout,
                    model_type=args.model,
                    resolution_values=args.resolution_values,
                    volume_change_threshold=args.volume_change_threshold,
                    canonicalize=args.canonicalize,
                    hydroelastic=args.hydroelastic,
                    max_convex_hull=args.max_convex_hull,
                )

                # Copy generated output files to the target directory.
                copied_sdf_paths = copy_generated_outputs_to_target_dir(
                    original_dir=original_dir,
                    asset_output_dir=asset_output_dir,
                    simulatable_sdf_paths=simulatable_sdf_paths,
                )

                logger.info(
                    f"Successfully processed {asset_name}. Output files: "
                    f"{copied_sdf_paths}"
                )

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user (Ctrl+C)")
            raise  # Re-raise to allow proper termination
        except Exception as e:
            logger.error(f"Failed to process {asset_path}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            continue


if __name__ == "__main__":
    args = parse_args()

    # Create output directory first to ensure it exists for logging.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to write to a file in the output directory.
    log_level = logging.INFO if args.debug else logging.WARNING
    log_file = output_dir / "make_asset_drake_compatible.log"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",  # Overwrite log file on each run
    )

    main(args)
