import os
import re
import random
from typing import List, Tuple


def get_original_image_id(filename: str) -> str:
    """
    Extract the original image ID from a cropped filename.

    Examples:
        IMG_1225_crop_1.jpg -> IMG_1225
        IMG_1225_crop_2.jpg -> IMG_1225
        IMG_1300.jpg        -> IMG_1300
    """
    base_name = os.path.splitext(filename)[0]

    # Remove crop suffix such as _crop_1, _crop_2, etc.
    original_id = re.sub(r"_crop_\d+$", "", base_name)

    return original_id


def create_group_split(
    image_files: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices by original image ID.

    All crops derived from the same original image are assigned
    to the same subset to prevent crop-level data leakage.

    Parameters
    ----------
    image_files:
        List of image filenames in the same order used by the dataset.

    train_ratio:
        Proportion of original-image groups assigned to training.

    val_ratio:
        Proportion of original-image groups assigned to validation.

    seed:
        Fixed random seed for reproducibility.

    Returns
    -------
    train_indices, val_indices, test_indices
    """

    if not image_files:
        raise ValueError("image_files is empty.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1 "
            "so that a test subset remains."
        )

    # -------------------------------------------------
    # Group sample indices by original image ID
    # -------------------------------------------------
    groups = {}

    for index, filename in enumerate(image_files):
        original_id = get_original_image_id(filename)

        if original_id not in groups:
            groups[original_id] = []

        groups[original_id].append(index)

    # -------------------------------------------------
    # Shuffle original-image groups
    # -------------------------------------------------
    group_ids = list(groups.keys())
    random.Random(seed).shuffle(group_ids)

    n_groups = len(group_ids)

    n_train_groups = int(train_ratio * n_groups)
    n_val_groups = int(val_ratio * n_groups)

    train_group_ids = group_ids[:n_train_groups]

    val_group_ids = group_ids[
        n_train_groups:
        n_train_groups + n_val_groups
    ]

    test_group_ids = group_ids[
        n_train_groups + n_val_groups:
    ]

    # -------------------------------------------------
    # Convert group IDs back into sample indices
    # -------------------------------------------------
    def collect_indices(selected_group_ids: List[str]) -> List[int]:
        indices = []

        for group_id in selected_group_ids:
            indices.extend(groups[group_id])

        return indices

    train_indices = collect_indices(train_group_ids)
    val_indices = collect_indices(val_group_ids)
    test_indices = collect_indices(test_group_ids)

    # -------------------------------------------------
    # Safety checks
    # -------------------------------------------------
    train_groups = set(train_group_ids)
    val_groups = set(val_group_ids)
    test_groups = set(test_group_ids)

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    all_indices = (
        train_indices
        + val_indices
        + test_indices
    )

    assert len(all_indices) == len(image_files)
    assert len(set(all_indices)) == len(image_files)

    # -------------------------------------------------
    # Print split summary
    # -------------------------------------------------
    print("\n========== GROUP-BASED DATASET SPLIT ==========")

    print(f"Original image groups: {n_groups}")

    print(
        f"Training groups      : {len(train_group_ids)}"
    )

    print(
        f"Validation groups    : {len(val_group_ids)}"
    )

    print(
        f"Testing groups       : {len(test_group_ids)}"
    )

    print(
        f"Training samples     : {len(train_indices)}"
    )

    print(
        f"Validation samples   : {len(val_indices)}"
    )

    print(
        f"Testing samples      : {len(test_indices)}"
    )

    return train_indices, val_indices, test_indices