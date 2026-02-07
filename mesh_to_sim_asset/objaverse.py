import logging
import multiprocessing
import os
import random
import shutil
import time
import urllib.error

from typing import Any, Hashable

import objaverse as ox
import objaverse.xl as oxl
import pandas as pd

from datasets import Dataset, load_dataset
from tqdm import tqdm

# Set up logger.
logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    success_delay: float = 0.0,
) -> Any:
    """
    Retry a function with exponential backoff on rate limiting errors.

    Args:
        func: The function to retry. Should be a callable that takes no arguments.
        max_retries: The maximum number of retries to attempt before giving up.
        base_delay: The base delay in seconds for exponential backoff on retries.
        max_delay: The maximum delay in seconds to cap exponential backoff.
        success_delay: The delay in seconds to wait after a successful request
            to avoid overwhelming the API.

    Returns:
        The result of the successful function call.

    Raises:
        Exception: If all retry attempts are exhausted or if a non-rate-limiting
            error occurs.
    """
    for attempt in range(max_retries + 1):
        try:
            result = func()
            # Add delay after successful request if specified.
            if success_delay > 0:
                time.sleep(success_delay)
            return result
        except Exception as e:
            # Check if this is a retryable error.
            is_rate_limit_error = (
                (isinstance(e, urllib.error.HTTPError) and e.code == 429)
                or "429" in str(e)
                or "Too Many Requests" in str(e)
            )
            is_network_error = (
                isinstance(e, urllib.error.ContentTooShortError)
                or isinstance(e, urllib.error.URLError)
                or "Connection" in str(e)
                or "timeout" in str(e).lower()
                or "network" in str(e).lower()
            )

            is_retryable = is_rate_limit_error or is_network_error

            if not is_retryable or attempt == max_retries:
                # Not a retryable error or we've exhausted retries.
                raise e

            # Calculate delay with exponential backoff.
            delay = min(base_delay * (2**attempt), max_delay)

            error_type = "Rate limited" if is_rate_limit_error else "Network error"
            logger.warning(
                f"{error_type} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.1f} seconds..."
            )
            time.sleep(delay)


def handle_new_object(
    local_path: str, file_identifier: str, sha256: str, metadata: dict[Hashable, Any]
) -> None:
    """
    Handle a newly downloaded object by deleting it to save disk space.

    Args:
        local_path: Path to the downloaded object file.
        file_identifier: Unique identifier for the file.
        sha256: SHA256 hash of the file.
        metadata: Dictionary containing metadata about the object.
    """
    # Delete all new objects to save disk space.
    try:
        os.remove(local_path)
    except OSError as e:
        logger.error(f"Error deleting {local_path}: {e}")


def download_objaverse_1_objects_with_batching(
    uuids: list[str],
    download_processes: int = 1,
    batch_size: int = 10,
    max_retries: int = 5,
    base_delay: float = 1.0,
    batch_delay: float = 0.5,
) -> dict[str, str]:
    """
    Download Objaverse 1.0 objects with batching to avoid rate limiting.

    Args:
        uuids: List of UIDs to download.
        download_processes: Number of processes for downloading.
        batch_size: Number of objects to download in each batch.
        max_retries: Maximum number of retry attempts per batch.
        base_delay: Base delay for exponential backoff on retries.
        batch_delay: Delay between successful batches.
        download_processes: Number of processes for downloading.

    Returns:
        Dictionary mapping UUIDs to object paths.
    """
    all_objects = {}
    total_batches = (len(uuids) + batch_size - 1) // batch_size
    skipped_batches = 0

    for i in tqdm(
        range(0, len(uuids), batch_size),
        desc="Downloading Objaverse 1.0 batches",
        unit="batch",
    ):
        batch = uuids[i : i + batch_size]
        batch_num = i // batch_size + 1

        def download_batch():
            return ox.load_objects(uids=batch, download_processes=download_processes)

        try:
            batch_objects = retry_with_backoff(
                download_batch,
                max_retries=max_retries,
                base_delay=base_delay,
                success_delay=batch_delay if batch_num < total_batches else 0,
            )
            all_objects.update(batch_objects)
        except Exception as e:
            # Continue to the next batch instead of failing.
            skipped_batches += 1
            logger.warning(
                f"Skipping batch {batch_num}/{total_batches} due to error: {e}. "
                f"This batch contained {len(batch)} objects."
            )

    logger.info(
        f"Downloaded {len(all_objects)} total objects across "
        f"{total_batches - skipped_batches}/{total_batches} successful batches"
    )
    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} problematic batches")
    return all_objects


def download_objaverse_xl_objects_with_batching(
    annotations: pd.DataFrame,
    download_dir: str,
    download_processes: int = 1,
    batch_size: int = 10,
    max_retries: int = 5,
    base_delay: float = 1.0,
    batch_delay: float = 0.5,
) -> dict[str, str]:
    """
    Download Objaverse XL objects with batching to avoid rate limiting.

    Args:
        annotations: DataFrame of annotations for objects to download.
        download_dir: The directory to download the dataset to.
        download_processes: Number of processes for downloading.
        batch_size: Number of objects to download in each batch.
        max_retries: Maximum number of retry attempts per batch.
        base_delay: Base delay for exponential backoff on retries.
        batch_delay: Delay between successful batches.

    Returns:
        Dictionary mapping file identifiers to object paths.
    """
    all_objects = {}
    total_batches = (len(annotations) + batch_size - 1) // batch_size
    skipped_batches = 0

    for i in tqdm(
        range(0, len(annotations), batch_size),
        desc="Downloading Objaverse XL batches",
        unit="batch",
    ):
        batch_annotations = annotations.iloc[i : i + batch_size]
        batch_num = i // batch_size + 1

        def download_batch():
            return oxl.download_objects(
                objects=batch_annotations,
                download_dir=download_dir,
                processes=download_processes,
                handle_found_object=None,
                handle_modified_object=None,
                handle_missing_object=None,
                save_repo_format="files",
                handle_new_object=handle_new_object,
            )

        try:
            batch_objects = retry_with_backoff(
                download_batch,
                max_retries=max_retries,
                base_delay=base_delay,
                success_delay=batch_delay if batch_num < total_batches else 0,
            )
            all_objects.update(batch_objects)
        except Exception as e:
            # Continue to the next batch instead of failing.
            skipped_batches += 1
            logger.warning(
                f"Skipping batch {batch_num}/{total_batches} due to error: {e}. "
                f"This batch contained {len(batch_annotations)} objects."
            )

    logger.info(
        f"Downloaded {len(all_objects)} total Objaverse XL objects across "
        f"{total_batches - skipped_batches}/{total_batches} successful batches"
    )
    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} problematic batches")
    return all_objects


def download_objaverse_xl(
    download_dir: str,
    download_alignment_annotations: bool,
    exclude_objaverse_1: bool,
    num_samples: int | None = None,
    download_processes: int = multiprocessing.cpu_count(),
    batch_size: int = 10,
    max_retries: int = 5,
    batch_delay: float = 0.5,
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Download the Objaverse XL dataset.

    Args:
        download_dir: The directory to download the dataset to.
        download_alignment_annotations: Whether to download alignment annotations for a
            higher-quality subset of the dataset. If False, downloads annotations for
            the entire dataset.
        exclude_objaverse_1: Whether to exclude the Objaverse 1 dataset (subset of
            Objaverse XL).
        num_samples: The number of samples to download. If None, all samples are
            downloaded.
        download_processes: The number of processes to use for downloading.
        batch_size: Number of objects to download per batch.
        max_retries: Maximum retry attempts per batch.
        batch_delay: Delay in seconds between successful batches.

    Returns:
        - file_to_path_map: A map from file identifier to the path of the object.
        - annotations: The annotations for the Objaverse XL dataset.
    """
    if download_alignment_annotations:
        # Download annotations for a higher-quality subset of the dataset.
        logger.info("Loading Objaverse XL alignment annotations with retry logic...")
        annotations = retry_with_backoff(
            lambda: oxl.get_alignment_annotations(download_dir=download_dir)
        )
        logger.info(f"Downloaded {len(annotations)} alignment annotations.")
    else:
        # Download the annotations for the entire dataset.
        logger.info("Loading Objaverse XL annotations with retry logic...")
        annotations = retry_with_backoff(
            lambda: oxl.get_annotations(download_dir=download_dir)
        )
        logger.info(f"Downloaded {len(annotations)} annotations.")

    if exclude_objaverse_1:
        annotations = annotations[annotations["source"] != "sketchfab"]

    if num_samples is not None:
        # Randomly sample num_samples objects.
        annotations = annotations.sample(num_samples)
        logger.info(f"Sampled {len(annotations)} Objaverse XL objects.")

    # Download objects with batching and retry logic.
    logger.info("Downloading Objaverse XL objects with batching and retry logic...")
    file_to_path_map = download_objaverse_xl_objects_with_batching(
        annotations=annotations,
        download_dir=download_dir,
        download_processes=download_processes,
        batch_size=batch_size,
        max_retries=max_retries,
        batch_delay=batch_delay,
    )
    logger.info(f"Downloaded {len(file_to_path_map)} Objaverse XL objects.")

    return file_to_path_map, annotations


def download_objaverse_plus_plus(
    min_objaverse_plus_plus_score: int | None = None,
    download_processes: int = multiprocessing.cpu_count(),
    batch_size: int = 10,
    max_retries: int = 5,
    batch_delay: float = 0.5,
) -> tuple[Dataset, pd.DataFrame, dict[str, str]]:
    """
    Download the Objaverse++ dataset.

    Args:
        min_objaverse_plus_plus_score: The minimum score for the Objaverse++ dataset.
            Scores are int in [0,3]. If None, no score filtering is applied.
        download_processes: The number of processes to use for downloading.
        batch_size: Number of objects to download per batch.
        max_retries: Maximum retry attempts per batch.
        batch_delay: Delay in seconds between successful batches.

    Returns:
        - objaverse_plus_plus: The Objaverse++ dataset.
        - annotations: The annotations for the Objaverse++ dataset.
        - uid_to_path_map: A map from UID to the path of the object.
    """
    objaverse_plus_plus = load_dataset("cindyxl/ObjaversePlusPlus", split="train")

    # Filter by quality.
    objaverse_plus_plus = objaverse_plus_plus.filter(
        lambda x: x["is_multi_object"] == "false"
    )
    objaverse_plus_plus = objaverse_plus_plus.filter(
        lambda x: x["is_single_color"] == "false"
    )
    if min_objaverse_plus_plus_score is not None:
        objaverse_plus_plus = objaverse_plus_plus.filter(
            lambda x: x["score"] >= min_objaverse_plus_plus_score
        )

    uids = objaverse_plus_plus["UID"]
    logger.info("Loading Objaverse++ annotations with retry logic...")
    annotations = retry_with_backoff(lambda: ox.load_annotations(uids=uids))
    logger.info("Loading Objaverse++ objects with batching and retry logic...")
    uid_to_path_map = download_objaverse_1_objects_with_batching(
        uuids=uids,
        batch_size=batch_size,
        max_retries=max_retries,
        batch_delay=batch_delay,
        download_processes=download_processes,
    )
    logger.info(f"Downloaded {len(objaverse_plus_plus)} Objaverse++ objects.")
    return objaverse_plus_plus, annotations, uid_to_path_map


def download_bs_objaverse(
    min_bs_objaverse_score: int | None = None,
    only_photorealistic: bool = False,
    download_processes: int = multiprocessing.cpu_count(),
    batch_size: int = 10,
    max_retries: int = 5,
    batch_delay: float = 0.5,
) -> tuple[Dataset, pd.DataFrame, dict[str, str]]:
    """
    Download the BS-Objaverse dataset.

    Args:
        min_bs_objaverse_score: The minimum score for the BS-Objaverse dataset.
            Scores are int in [1,6]. If None, no score filtering is applied.
        only_photorealistic: Whether to only download the photorealistic objects
            from the BS-Objaverse dataset.
        download_processes: The number of processes to use for downloading.
        batch_size: Number of objects to download per batch.
        max_retries: Maximum retry attempts per batch.
        batch_delay: Delay in seconds between successful batches.

    Returns:
        - bs_objaverse: The BS-Objaverse dataset.
        - annotations: The annotations for the BS-Objaverse dataset.
        - uid_to_path_map: A map from UID to the path of the object.
    """
    bs_objaverse = load_dataset("Zery/BS-Objaverse", name="BS-Objaverse", split="train")

    # Filter by quality.
    if min_bs_objaverse_score is not None:
        bs_objaverse = bs_objaverse.filter(
            lambda x: x["score"] >= min_bs_objaverse_score
        )
    if only_photorealistic:
        bs_objaverse = bs_objaverse.filter(lambda x: x["style"] == "Photo_realistic")

    uids = bs_objaverse["id"]
    logger.info("Loading BS-Objaverse annotations with retry logic...")
    annotations = retry_with_backoff(lambda: ox.load_annotations(uids=uids))
    logger.info("Loading BS-Objaverse objects with batching and retry logic...")
    uid_to_path_map = download_objaverse_1_objects_with_batching(
        uuids=uids,
        batch_size=batch_size,
        max_retries=max_retries,
        batch_delay=batch_delay,
        download_processes=download_processes,
    )
    logger.info(f"Downloaded {len(bs_objaverse)} BS-Objaverse objects.")
    return bs_objaverse, annotations, uid_to_path_map


def download_high_quality_objaverse_1(
    min_bs_objaverse_score: int = 4,
    min_objaverse_plus_plus_score: int = 2,
    num_samples: int | None = None,
    download_dir: str | None = None,
    download_processes: int = multiprocessing.cpu_count(),
    batch_size: int = 10,
    max_retries: int = 5,
    batch_delay: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Download the high-quality subset of the Objaverse 1.0 dataset based on both the
    BS-Objaverse and Objaverse++ datasets.

    Args:
        min_bs_objaverse_score: The minimum score for the BS-Objaverse dataset.
            Scores are int in [1,6]. BS-Objaverse uses scores >=4 for their models.
        min_objaverse_plus_plus_score: The minimum score for the Objaverse++ dataset.
            Scores are int in [0,3]. Objaverse++ uses scores >=2 for their models.
        num_samples: The number of samples to download. If None, all samples are
            downloaded.
        download_dir: The directory to copy the downloaded objects to. If None, the
            objects are not copied from the default ~/.objaverse location.
        download_processes: The number of processes to use for downloading.
        batch_size: Number of objects to download per batch.
        max_retries: Maximum retry attempts per batch.
        batch_delay: Delay in seconds between successful batches.

    Returns:
        - annotations: The annotations for the Objaverse 1.0 dataset.
        - uid_to_path_map: A map from UID to the path of the object.
    """
    # Download and filter the BS-Objaverse dataset.
    bs_objaverse = load_dataset("Zery/BS-Objaverse", name="BS-Objaverse", split="train")
    bs_objaverse = bs_objaverse.filter(lambda x: x["score"] >= min_bs_objaverse_score)
    bs_objaverse = bs_objaverse.filter(lambda x: x["style"] == "Photo_realistic")
    bs_objaverse_uids = bs_objaverse["id"]

    # Download and filter the Objaverse++ dataset.
    objaverse_plus_plus = load_dataset("cindyxl/ObjaversePlusPlus", split="train")
    objaverse_plus_plus = objaverse_plus_plus.filter(
        lambda x: x["score"] >= min_objaverse_plus_plus_score
    )
    objaverse_plus_plus = objaverse_plus_plus.filter(
        lambda x: x["is_multi_object"] == "false"
    )
    objaverse_plus_plus = objaverse_plus_plus.filter(
        lambda x: x["is_single_color"] == "false"
    )
    objaverse_plus_plus_uids = objaverse_plus_plus["UID"]

    # Take the intersection of the two datasets.
    objaverse_uids = list(set(bs_objaverse_uids) & set(objaverse_plus_plus_uids))
    logger.info(f"Selected {len(objaverse_uids)} high-quality Objaverse 1.0 objects.")

    if num_samples is not None:
        # Randomly sample num_samples objects.
        objaverse_uids = random.sample(objaverse_uids, num_samples)
        logger.info(f"Sampled {len(objaverse_uids)} Objaverse 1.0 objects.")

    # Download the objects with retry logic for annotations and batching for objects.
    logger.info("Loading annotations with retry logic...")
    annotations = retry_with_backoff(lambda: ox.load_annotations(uids=objaverse_uids))
    logger.info("Loading objects with batching and retry logic...")
    uid_to_path_map = download_objaverse_1_objects_with_batching(
        uuids=objaverse_uids,
        batch_size=batch_size,
        max_retries=max_retries,
        batch_delay=batch_delay,
        download_processes=download_processes,
    )
    logger.info(f"Downloaded {len(uid_to_path_map)} Objaverse 1.0 objects.")

    if download_dir is not None:
        # Copy the objects and modify the paths.
        objaverse_home = os.path.expanduser("~/.objaverse")
        objaverse_v1_dir = os.path.join(download_dir, "objaverse_v1")

        # Copy the entire ~/.objaverse directory to download_dir/objaverse_v1.
        logger.info(f"Copying {objaverse_home} to {objaverse_v1_dir}")
        if os.path.exists(objaverse_v1_dir):
            shutil.rmtree(objaverse_v1_dir)
        shutil.copytree(objaverse_home, objaverse_v1_dir)
        logger.info(f"Copied {objaverse_home} to {objaverse_v1_dir}")

        # Update all paths in uid_to_path_map to point to the new location.
        updated_uid_to_path_map = {}
        for uid, old_path in uid_to_path_map.items():
            # Replace the ~/.objaverse prefix with the new objaverse_v1 directory.
            if old_path.startswith(objaverse_home):
                new_path = old_path.replace(objaverse_home, objaverse_v1_dir)
                updated_uid_to_path_map[uid] = new_path
            else:
                updated_uid_to_path_map[uid] = old_path

        uid_to_path_map = updated_uid_to_path_map
        logger.info(
            f"Updated {len(uid_to_path_map)} paths to point to {objaverse_v1_dir}"
        )

    return annotations, uid_to_path_map
