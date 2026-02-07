import argparse
import logging

from mesh_to_sim_asset.objaverse import (
    download_bs_objaverse,
    download_objaverse_plus_plus,
    download_objaverse_xl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dir",
        type=str,
        default="~/.objaverse",
        help="The directory to download the dataset to.",
    )
    parser.add_argument(
        "--download_objaverse_xl",
        action="store_true",
        help="Download the Objaverse XL dataset.",
    )
    parser.add_argument(
        "--download_objaverse_plus_plus",
        action="store_true",
        help="Download the Objaverse++ dataset.",
    )
    parser.add_argument(
        "--download_bs_objaverse",
        action="store_true",
        help="Download the BS-Objaverse dataset.",
    )
    parser.add_argument(
        "--download_alignment_annotations",
        action="store_true",
        help="Download the alignment annotations for the Objaverse XL dataset. "
        "Otherwise, download the annotations for the entire dataset.",
    )
    parser.add_argument(
        "--min_bs_objaverse_score",
        type=float,
        help="The minimum score for the BS-Objaverse dataset. Scores are int in [1,6].",
    )
    parser.add_argument(
        "--min_objaverse_plus_plus_score",
        type=float,
        help="The minimum score for the Objaverse++ dataset. Scores are int in [0,3].",
    )
    parser.add_argument(
        "--only_photorealistic",
        action="store_true",
        help="Only download the photorealistic objects from the BS-Objaverse dataset.",
    )
    parser.add_argument(
        "--exclude_objaverse_1",
        action="store_true",
        help="Exclude the Objaverse 1 dataset (subset of Objaverse XL).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging.
    log_level = logging.INFO if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.download_objaverse_xl:
        file_to_path_map, annotations = download_objaverse_xl(
            download_dir=args.download_dir,
            download_alignment_annotations=args.download_alignment_annotations,
            exclude_objaverse_1=args.exclude_objaverse_1,
        )
    if args.download_objaverse_plus_plus:
        objaverse_plus_plus, annotations, uid_to_path_map = (
            download_objaverse_plus_plus(
                min_objaverse_plus_plus_score=args.min_objaverse_plus_plus_score,
            )
        )
    if args.download_bs_objaverse:
        bs_objaverse, annotations, uid_to_path_map = download_bs_objaverse(
            min_bs_objaverse_score=args.min_bs_objaverse_score,
            only_photorealistic=args.only_photorealistic,
        )


if __name__ == "__main__":
    main()
