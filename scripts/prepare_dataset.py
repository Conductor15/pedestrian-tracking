from src.preprocess_mot17 import (
    process_all_folders,
    move_files_all_folders,
    delete_subfolders,
)
import shutil
import os


def prepare_split(split_path):
    """
    Run the full preprocessing pipeline for a dataset split.

    Steps:
        1. Filter FRCNN sequences and generate YOLO labels
        2. Collect images and labels into unified folders
        3. Remove intermediate sequence folders

    Args:
        split_path (str): Path to dataset split (train or test).
    """

    print(f"\nProcessing dataset split: {split_path}")

    process_all_folders(split_path)
    move_files_all_folders(split_path)
    delete_subfolders(split_path)


def main():
    raw_root = "data/raw/MOT17"
    processed_root = "data/processed/mot17"

    train_raw = os.path.join(raw_root, "train")
    test_raw = os.path.join(raw_root, "test")

    train_processed = os.path.join(processed_root, "train")
    test_processed = os.path.join(processed_root, "test")

    # Copy raw → processed
    if not os.path.exists(train_processed):
        shutil.copytree(train_raw, train_processed)

    if not os.path.exists(test_processed):
        shutil.copytree(test_raw, test_processed)

    prepare_split(train_processed)
    prepare_split(test_processed)

    print("\nDataset preprocessing completed.")


if __name__ == "__main__":
    main()