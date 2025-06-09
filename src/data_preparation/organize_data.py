from src.data_preparation.constants import DATA_FOLDER

import shutil
import os
import logging
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
import random

random.seed(1)


def copy_images_to_joined_folder(folder_name: str) -> None:
    """
    Copy images from the processed folder to the joined folder.

    Parameters:
    - folder_name (str): Folder name for saving the processed data.
    """
    source_folder = DATA_FOLDER / "torch" / folder_name
    target_folder = DATA_FOLDER / "torch" / "joined"
    target_folder.mkdir(parents=True, exist_ok=True)

    for coasttype in os.listdir(source_folder):
        source_path = source_folder / coasttype
        destination_path = target_folder / coasttype
        destination_path.mkdir(parents=True, exist_ok=True)
        for img_file in os.listdir(source_path):
            shutil.copy(source_path / img_file, destination_path / img_file)


def split_single_coasttype(source_path: Path, train_path: Path, val_path: Path, ratio: float = 0.8) -> None:
    """
    Split images of a single coastal type into training and validation sets.
    Parameters:
    - source_path (Path): Path to the folder containing images of a single coastal type.
    - train_path (Path): Path to the folder where training images will be saved.
    - val_path (Path): Path to the folder where validation images will be saved.
    - ratio (float): Ratio of training to validation images. Default is 0.8.
    """
    img_files = os.listdir(source_path)
    random.shuffle(img_files)
    split_index = int(len(img_files) * ratio)

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for img_file in img_files[:split_index]:
        shutil.move(source_path / img_file, train_path / img_file)
    for img_file in img_files[split_index:]:
        shutil.move(source_path / img_file, val_path / img_file)
    shutil.rmtree(source_path)


def split_images(folder_path: Path, ratio: float = 0.8) -> None:
    """
    Split images into training and validation sets for all coastal types.
    Parameters:
    - folder_path (Path): Path to the folder containing the images.
    - ratio (float): Ratio of training to validation images. Default is 0.8.
    """
    train_folder = folder_path / "train"
    val_folder = folder_path / "test"
    train_folder.mkdir(parents=True, exist_ok=True)
    val_folder.mkdir(parents=True, exist_ok=True)

    coasttype_folders = [
        f for f in os.listdir(folder_path) if (folder_path / f).is_dir() and f not in ["train", "test"]
    ]

    for coasttype in tqdm(coasttype_folders, desc="Splitting images into training and test sets."):
        source_path = folder_path / coasttype
        train_path = train_folder / coasttype
        val_path = val_folder / coasttype
        split_single_coasttype(source_path, train_path, val_path, ratio)

    logging.info("Images split into training and test sets successfully.")
    log_summary(folder_path)


def log_summary(folder_path: Path) -> None:
    """
    Log a summary of the dataset sizes.
    Parameters:
    - folder_path (Path): Path to the folder containing the images.
    """
    table = []
    train_root, test_root = folder_path / "train", folder_path / "test"
    total_train, total_test = 0, 0
    coastal_types = sorted(os.listdir(train_root))
    for coastal_type in coastal_types:
        train_count = len(os.listdir(train_root / coastal_type)) if (train_root / coastal_type).exists() else 0
        test_count = len(os.listdir(test_root / coastal_type)) if (test_root / coastal_type).exists() else 0
        table.append([coastal_type, train_count, test_count])

        total_train += train_count
        total_test += test_count

    table.append(["**Total**", total_train, total_test])

    headers = ["Coastal Type", "Train Size", "Test Size"]
    table_str = tabulate(table, headers, tablefmt="github")
    logging.info("\n" + table_str)


def main() -> None:
    """
    Main function to organize images for DataLoader.
    """
    folders = os.listdir(DATA_FOLDER / "torch")
    for folder_name in tqdm(folders):
        logging.info(f"Copying {folder_name} images to the joined folder.")
        copy_images_to_joined_folder(folder_name)
    logging.info("Images copied successfully.")

    split_images(DATA_FOLDER / "torch" / "joined")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
