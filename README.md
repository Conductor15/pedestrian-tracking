# Pedestrian Tracking with YOLOv8 and DeepSORT

## Demo
Pedestrian tracking using YOLOv8 detector and DeepSORT tracker.
![Tracking Demo](demo/demo_1s.gif)

Full video: [output_video.avi](demo/demo_1s.avi)

## Overview
This project implements a multi-object pedestrian tracking system that combines a deep learning–based object detector with an appearance-based tracking algorithm.

Pedestrians are detected using Ultralytics YOLOv8 and tracked across frames using DeepSORT.
The system assigns persistent IDs to pedestrians and maintains identity consistency even when objects move across frames or experience short-term occlusion.

Such tracking systems are widely used in:

    - intelligent surveillance systems

    - crowd analysis

    - smart city monitoring

    - autonomous driving perception

    - retail analytics

## Project Structure
```
    pedestrian-tracking
    │
    ├── configs/               # Configuration files
    │
    ├── data/
    │   ├── raw/               # Raw dataset
    │   └── processed/         # Processed dataset (YOLO format)
    │
    ├── deep_sort/             # DeepSORT tracking
    │
    ├── demo/                  # Demo videos and GIFs
    │
    ├── models/                # Trained detector and ReID weights
    │
    ├── notebooks/             # Experiments and analysis
    │
    ├── scripts/               # Training and inference scripts
    │   ├── prepare_dataset.py
    │   ├── train_detector.py
    │   ├── test_detector.py
    │   ├── demo_detector.py
    │   └── track_video.py
    │
    ├── src/                   # Core tracking pipeline
    │   ├── detector.py        # YOLOv8 pedestrian detector wrapper
    │   ├── tracker.py         # DeepSORT tracker interface
    │   ├── tracking_pipeline.py  # End-to-end tracking pipeline
    │   ├── preprocess_mot17.py   # MOT17 dataset preprocessing
    │   │
    │   └── utils/
    │       └── visualization.py  # Bounding box & ID visualization
    │
    ├── outputs/               # Tracking results
    ├── runs/                  # Training logs and checkpoints
    │
    ├── requirements.txt
    ├── Makefile
    └── README.md
```

## Dataset
This project uses the MOT17 dataset, a widely used benchmark for multi-object pedestrian tracking.

The dataset contains video sequences with frame-level bounding box annotations for pedestrians, making it suitable for training pedestrian detectors and evaluating tracking algorithms.

Due to its large size, the dataset is not included in this repository.

## Installation
## Installation

Follow the steps below to set up the project environment.

### 1. Clone the repository

```bash
git clone https://github.com/Conductor15/pedestrian-tracking.git
cd pedestrian-tracking
```

### 2. Install dependencies

Install the required Python packages:
```bash
make install
```
After installation, the environment will contain all dependencies required for training the detector and running the tracking pipeline.

## Data Preparation
### 1. Download

You can download the dataset from Kaggle:

 https://www.kaggle.com/datasets/wenhoujinjust/mot-17

After downloading, extract the dataset and place it in `data/raw/`

Detailed instructions for dataset organization can be found in `data/raw/notes.md`
### 2. Processing

Once the dataset is placed in the correct directory, run the preprocessing pipeline:
```bash
make process_data
```
This step will:

- parse the MOT17 annotations

- convert bounding boxes to YOLO format

- generate training and validation splits

- store the processed dataset in `data/processed/`

## Training Detector

The pedestrian detector is trained using Ultralytics YOLOv8 on the processed dataset.

Before training, ensure that the dataset has been prepared and converted to the required format.

Run the following command to start training:
```bash
make train_detector
```
During training, the system will:

load the processed dataset from `data/processed/`

train the YOLOv8 pedestrian detector

save training logs and checkpoints

Training outputs are stored in `runs/`

This directory contains:

- training logs

- model checkpoints

- evaluation metrics

The trained detector weights will be saved in the `models/` directory.

## Tracking Pipeline
The tracking pipeline combines object detection with multi-object tracking to assign consistent IDs to pedestrians across video frames.

This project uses:

Ultralytics YOLOv8 for pedestrian detection

DeepSORT for multi-object tracking

To run the full tracking pipeline on a video:
```bash
make track_video
```

The pipeline performs the following steps:

- Detect pedestrians in each frame using YOLOv8

- Extract appearance embeddings for each detection

- Associate detections across frames using DeepSORT

- Assign and maintain unique tracking IDs

- Tracking results will be saved in `outputs/`
## Results
### 1. Detector Performance

The pedestrian detector was trained and evaluated using Ultralytics YOLOv8 on the MOT17 dataset.

Evaluation results on the validation set:

Metric	        Score
Precision	    0.903
Recall	        0.803
mAP@0.5	        0.901
mAP@0.5:0.95	0.652

The detector achieves strong performance for pedestrian detection in crowded scenes, which provides reliable inputs for the multi-object tracking pipeline.
### 2.Tracking Demo

The tracking pipeline combines the YOLOv8 detector with DeepSORT to maintain consistent pedestrian identities across frames.

- The output video contains:

- pedestrian bounding boxes

- unique tracking IDs

- consistent identity tracking across frames

Full video output: `demo/output_video.avi`
## Acknowledgements
This project builds upon several open-source works in the fields of object detection and multi-object tracking.

Ultralytics YOLOv8 for pedestrian detection

DeepSORT for multi-object tracking. Implementation reference: https://github.com/nwojke/deep_sort

We thank the authors for making their work publicly available.

## License
This project is released under the MIT License.