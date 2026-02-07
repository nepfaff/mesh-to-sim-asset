import argparse
import hashlib
import json
import logging
import multiprocessing
import random
import shutil

from pathlib import Path

from tqdm import tqdm

from mesh_to_sim_asset.create_drake_assets_from_geometry import (
    process_meshes,
    should_skip_existing,
)
from mesh_to_sim_asset.objaverse import (
    download_high_quality_objaverse_1,
    download_objaverse_xl,
)
from mesh_to_sim_asset.util import (
    CollisionGeomType,
    OpenAIModelType,
    get_basename_without_extension,
)

# Set up logger.
logger = logging.getLogger(__name__)


def rename_downloaded_meshes(
    uid_to_path_map_objaverse_1: dict[str, str],
    file_to_path_map_objaverse_xl: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Rename downloaded mesh files to avoid naming conflicts between datasets.

    Args:
        uid_to_path_map_objaverse_1: Mapping from UID to path for Objaverse 1.0.
        file_to_path_map_objaverse_xl: Mapping from fileIdentifier to path for
            Objaverse XL.

    Returns:
        Tuple of (renamed_uid_to_path_map, renamed_file_to_path_map,
                 uid_to_original_filename_map, file_to_original_filename_map)
    """
    renamed_uid_to_path_map = {}
    renamed_file_to_path_map = {}
    uid_to_original_filename_map = {}
    file_to_original_filename_map = {}

    # Rename Objaverse 1.0 meshes: `UID.ext`.
    logger.info("Renaming Objaverse 1.0 meshes to avoid conflicts...")
    for uid, original_path in tqdm(
        uid_to_path_map_objaverse_1.items(), desc="Renaming Objaverse 1.0"
    ):
        original_path = Path(original_path)
        if not original_path.exists():
            logger.warning(f"Original path does not exist: {original_path}")
            continue

        # Store original filename for metadata.
        uid_to_original_filename_map[uid] = original_path.name

        # Create new name: `UID.ext`.
        original_suffix = original_path.suffix
        new_name = f"{uid}{original_suffix}"
        new_path = original_path.parent / new_name

        # Check if file already has the target name.
        if original_path == new_path:
            renamed_uid_to_path_map[uid] = str(original_path)
            logger.debug(f"File already has target name: {original_path.name}")
            continue

        # Copy file to new location.
        try:
            shutil.copy2(original_path, new_path)
            renamed_uid_to_path_map[uid] = str(new_path)
            logger.debug(f"Renamed {original_path.name} -> {new_path.name}")
        except Exception as e:
            logger.error(f"Failed to rename {original_path} to {new_path}: {e}")
            # Keep original path if renaming fails.
            renamed_uid_to_path_map[uid] = str(original_path)

    # Rename Objaverse XL meshes: `fileIdentifier.ext`.
    logger.info("Renaming Objaverse XL meshes to avoid conflicts...")
    for file_id, original_path in tqdm(
        file_to_path_map_objaverse_xl.items(), desc="Renaming Objaverse XL"
    ):
        original_path = Path(original_path)
        if not original_path.exists():
            logger.warning(f"Original path does not exist: {original_path}")
            continue

        # Store original filename for metadata.
        file_to_original_filename_map[file_id] = original_path.name

        # Create a unique short filename using a hash of the fileIdentifier.
        # This ensures a consistent, short, and unique name for each file.
        hash_object = hashlib.md5(file_id.encode())
        short_filename = f"{hash_object.hexdigest()[:16]}{original_path.suffix}"
        new_path = original_path.parent / short_filename

        # Check if file already has the target name.
        if original_path == new_path:
            renamed_file_to_path_map[file_id] = str(original_path)
            logger.debug(f"File already has target name: {original_path.name}")
            continue

        # Copy file to new location.
        try:
            shutil.copy2(original_path, new_path)
            renamed_file_to_path_map[file_id] = str(new_path)
            logger.debug(f"Renamed {original_path.name} -> {new_path.name}")
        except Exception as e:
            logger.error(f"Failed to rename {original_path} to {new_path}: {e}")
            # Keep original path if renaming fails.
            renamed_file_to_path_map[file_id] = str(original_path)

    logger.info(f"Renamed {len(renamed_uid_to_path_map)} Objaverse 1.0 meshes")
    logger.info(f"Renamed {len(renamed_file_to_path_map)} Objaverse XL meshes")

    return (
        renamed_uid_to_path_map,
        renamed_file_to_path_map,
        uid_to_original_filename_map,
        file_to_original_filename_map,
    )


def download_and_get_path_to_metadata_map(args: argparse.Namespace) -> dict[str, str]:
    # Set random seed.
    random.seed(42)

    if args.num_samples is not None:
        # Split equally between both datasets.
        num_samples = args.num_samples // 2
    else:
        num_samples = None

    # Download the datasets.
    logger.info(f"Downloading Objaverse 1.0 dataset with {num_samples} samples.")
    annotations_objaverse_1, uid_to_path_map_objaverse_1 = (
        download_high_quality_objaverse_1(
            num_samples=num_samples,
            download_dir=args.download_dir,
            download_processes=args.download_processes,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            batch_delay=args.download_delay,
        )
    )
    logger.info(f"Downloading Objaverse XL dataset with {num_samples} samples.")
    file_to_path_map_objaverse_xl, annotations_objaverse_xl = download_objaverse_xl(
        download_dir=args.download_dir,
        download_alignment_annotations=True,  # Higher quality
        exclude_objaverse_1=True,  # Already downloaded
        num_samples=num_samples,
        download_processes=args.download_processes,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        batch_delay=args.download_delay,
    )

    # Rename downloaded meshes to avoid naming conflicts.
    (
        uid_to_path_map_objaverse_1,
        file_to_path_map_objaverse_xl,
        uid_to_original_filename_map,
        file_to_original_filename_map,
    ) = rename_downloaded_meshes(
        uid_to_path_map_objaverse_1, file_to_path_map_objaverse_xl
    )

    # Construct mapping from file path to metadata.
    path_to_metadata_map = {}
    for uid, path in uid_to_path_map_objaverse_1.items():
        path_to_metadata_map[path] = {
            "name": annotations_objaverse_1[uid]["name"],
            "categories": annotations_objaverse_1[uid]["categories"],
            "license": annotations_objaverse_1[uid]["license"],
            "dataset": "objaverse_1.0",
            "uid": uid,
            "original_filename": uid_to_original_filename_map[uid],
        }
    for uid, path in file_to_path_map_objaverse_xl.items():
        annotation = annotations_objaverse_xl[
            annotations_objaverse_xl["fileIdentifier"] == uid
        ].iloc[0]

        # Parse stringified metadata.
        raw_metadata = annotation["metadata"]
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            metadata = {}

        # Add additional fields.
        metadata["fileIdentifier"] = annotation["fileIdentifier"]
        metadata["license"] = annotation["license"]
        metadata["dataset"] = "objaverse_xl"
        metadata["original_filename"] = file_to_original_filename_map[uid]
        path_to_metadata_map[path] = metadata
    logging.info(f"Downloaded {len(path_to_metadata_map)} objects.")

    # Filter out meshes that should be skipped.
    if args.skip_existing:
        filtered_path_to_metadata_map = {}
        for path, metadata in path_to_metadata_map.items():
            if should_skip_existing(path, args.output_dir):
                mesh_name = get_basename_without_extension(Path(path))
                logger.info(
                    f"Skipping {mesh_name} because it already has an entry in the "
                    f"{args.output_dir} directory."
                )
            else:
                filtered_path_to_metadata_map[path] = metadata
        path_to_metadata_map = filtered_path_to_metadata_map
    if not path_to_metadata_map:
        logger.info("No meshes to process after filtering.")
        return

    # Create the metadata files.
    for path, metadata in tqdm(
        path_to_metadata_map.items(), desc="Creating metadata files"
    ):
        mesh_path = Path(path)
        metadata_path = mesh_path.with_name(f"{mesh_path.stem}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Created all metadata files.")

    return path_to_metadata_map


def main(args: argparse.Namespace):
    if args.path_to_metadata_map is not None:
        # Read the path to metadata map from a file.
        with open(args.path_to_metadata_map, "r", encoding="utf-8") as f:
            path_to_metadata_map = json.load(f)
    else:
        path_to_metadata_map = download_and_get_path_to_metadata_map(args)

        # Write the path to metadata map to a file.
        with open(
            Path(args.output_dir) / "path_to_metadata_map.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(path_to_metadata_map, f, indent=2, ensure_ascii=False)

    # Process the meshes.
    process_meshes(
        mesh_paths=list(path_to_metadata_map.keys()),
        output_dir=args.output_dir,
        is_metric=False,
        canonicalize=True,
        keep_images=args.keep_images,
        use_cpu_rendering=args.use_cpu_rendering,
        collision_geom_type=args.collision_geom_type,
        rolopoly_timeout=args.rolopoly_timeout,
        model_type=args.model,
        debug=args.debug,
        only_process_single_objects=True,
        only_process_textured=True,
        only_process_simulatable=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process Objaverse datasets into simulation-ready "
        "assets."
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        help="Total number of samples to download (will be split equally between "
        "Objaverse 1.0 and XL). If not specified, will download all available samples.",
    )
    parser.add_argument(
        "--download-dir",
        "-i",
        default="~/.objaverse",
        help="Directory to download the Objaverse datasets to (default: ~/.objaverse).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to store processed output files.",
    )
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Whether to skip meshes that already have an entry in the output "
        "directory.",
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
        "--collision-geom-type",
        "-g",
        type=CollisionGeomType,
        default=[CollisionGeomType.CoACD, CollisionGeomType.VTK],
        nargs="+",
        help="The type(s) of collision geometry to use for the mesh. Can specify "
        "multiple types.",
    )
    parser.add_argument(
        "--rolopoly-timeout",
        type=int,
        default=900,
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
        "--path-to-metadata-map",
        "-p",
        help="Path to the path to metadata map file. If provided, will skip the "
        "download and processing steps and only write the metadata map to a file.",
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use for downloading. Defaults to the "
        "number of CPU cores.",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between download requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of objects to download in each batch. Smaller batches are "
        "more resilient to rate limiting. Default is 10.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum number of retry attempts per batch when rate limited. "
        "Default is 10.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Create output directory first to ensure it exists for logging.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to write to a file in the output directory.
    log_level = logging.INFO if args.debug else logging.WARNING
    log_file = output_dir / "download_and_process_objaverse.log"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",  # Overwrite log file on each run
    )

    main(args)
