from src.tracking_pipeline import video_tracking
from src.detector import Detector
from src.tracker import deepSORT


def main():

    model_path = "models/detector/yolov8s_mot17.pt"
    video_path = "data/demo/demo_data3.mp4"

    detector = Detector(model_path)
    tracker = deepSORT()

    video_tracking(
        video_path,
        detector,
        tracker,
        is_save_result=True,
        save_dir="outputs"
    )


if __name__ == "__main__":
    main()