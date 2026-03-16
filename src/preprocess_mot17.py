import pandas as pd
import os
import shutil
import configparser

from tqdm import tqdm

def convert_to_yolo_format(bb, img_width, img_height):
    """
    Convert bounding box from MOT17 format to YOLO format.

    MOT17 format:
        (bb_left, bb_top, bb_width, bb_height)

    YOLO format:
        (x_center, y_center, width, height) normalized to [0, 1]

    Args:
        bb (dict or pandas.Series): Bounding box with keys
            'bb_left', 'bb_top', 'bb_width', 'bb_height'.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1].
    """
    x_center = bb['bb_left'] + bb['bb_width'] / 2
    y_center = bb['bb_top'] + bb['bb_height'] / 2

    x_center = x_center / img_width
    y_center = y_center / img_height
    bb_width = bb['bb_width'] / img_width
    bb_height = bb['bb_height'] / img_height

    x_center = max(min(x_center,1),0)
    y_center = max(min(y_center,1),0)
    bb_width = max(min(bb_width,1),0)
    bb_height = max(min(bb_height,1),0)

    return (x_center, y_center, bb_width, bb_height)


def process_folder(folder_path):
    """
    Convert MOT17 detection annotations in a sequence folder to YOLO label format.

    This function reads the detection file `det/det.txt` and image metadata from
    `seqinfo.ini`, converts bounding boxes from MOT format to YOLO format, and
    generates a label file for each frame.

    Args:
        folder_path (str): Path to a MOT17 sequence folder 
            (e.g., MOT17-02-FRCNN).

    Outputs:
        Creates a `labels/` directory inside the sequence folder containing
        YOLO-format label files for each frame.

    Notes:
        - MOT format: (bb_left, bb_top, bb_width, bb_height)
        - YOLO format: (x_center, y_center, width, height) normalized to [0, 1]
        - Each label line: "class_id x_center y_center width height"
    """
    
    config = configparser.ConfigParser()
    config.read(os.path.join(folder_path, 'seqinfo.ini'))
    img_width = int(config['Sequence']['imWidth'])
    img_height = int(config['Sequence']['imHeight'])

    det_path = os.path.join(folder_path, 'det/det.txt')
    df = pd.read_csv(
        det_path,
        header= None,
        names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    )

    labels_folder = os.path.join(folder_path, 'labels')
    os.makedirs(labels_folder, exist_ok = True)

    for frame_number in df['frame'].unique():
        frame_data = df[df['frame'] == frame_number]
        label_file = os.path.join(labels_folder, f'{frame_number:06d}.txt')

        with open(label_file, 'w') as file:
            for _, row in frame_data.iterrows():
                yolo_bb = convert_to_yolo_format(row, img_width, img_height)
                file.write(f'0 {yolo_bb[0]} {yolo_bb[1]} {yolo_bb[2]} {yolo_bb[3]}\n')
        

def process_all_folders(base_directory):
    """
    Process all MOT17 sequence folders inside a directory.

    This function iterates through all subfolders in the base directory,
    keeps only sequences containing 'FRCNN' in their names, and converts
    their detection annotations to YOLO format.

    Args:
        base_directory (str): Path to the MOT17 split directory
            (e.g., MOT17/train or MOT17/test).

    Notes:
        - Non-FRCNN sequences are removed.
        - Each valid sequence folder will be processed using `process_folder()`.
    """
    for folder_name in tqdm(os.listdir(base_directory)):
        folder_path = os.path.join(base_directory, folder_name)

        # Delete folder not contain 'FRCNN' in name
        if 'FRCNN' not in folder_name:
            os.system(f'rm -rf {folder_path}')
            continue

        if os.path.isdir(folder_path):
            process_folder(folder_path)
            

def rename_and_move_files(src_folder, dst_folder, folder_name, file_extension):
    """
    Move files from a source folder to a destination folder while renaming them.

    The sequence folder name is prefixed to each filename to avoid
    name collisions between different sequences.

    Args:
        src_folder (str): Folder containing source files.
        dst_folder (str): Destination folder where files will be moved.
        folder_name (str): Name of the sequence folder (used as prefix).
        file_extension (str): File extension filter (e.g., '.jpg', '.txt').
    """
    for filename in os.listdir(src_folder):
        if filename.endswith(file_extension):
            # Include folder name in the new filename
            new_filename = f'{folder_name}_{filename}'
            shutil.move(
                os.path.join(src_folder, filename),
                os.path.join(dst_folder, new_filename)
            )
            
            
def move_files_all_folders(base_directory):
    """
    Collect images and labels from all sequence folders into unified directories.

    This function moves:
        - images from `img1/`
        - labels from `labels/`

    into two global folders:
        images/
        labels/

    Args:
        base_directory (str): Path to processed MOT17 split directory.

    Notes:
        File names are prefixed with the sequence name to avoid duplicates.
    """
    images_dir = os.path.join(base_directory, 'images')
    labels_dir = os.path.join(base_directory, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for folder_name in tqdm(os.listdir(base_directory)):
        if folder_name in ['images', 'labels']:  # Skip these folders
            continue

        folder_path = os.path.join(base_directory, folder_name)

        if os.path.isdir(folder_path):
            rename_and_move_files(
                os.path.join(folder_path, 'img1'),
                images_dir,
                folder_name,
                '.jpg'
            )

            rename_and_move_files(
                os.path.join(folder_path, 'labels'),
                labels_dir,
                folder_name,
                '.txt'
            )
            
            
def delete_subfolders(base_directory):
    """
    Remove all sequence subfolders after files have been collected.

    Only the aggregated folders `images/` and `labels/` are preserved.

    Args:
        base_directory (str): Path to the dataset split directory.
    """
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        if os.path.isdir(folder_path) and folder_name not in ['images', 'labels']:
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_name}")