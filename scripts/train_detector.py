from ultralytics import YOLO
import shutil
import os

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Config
epochs = 1
batch_size = -1
img_size = 640
project_name = 'pedestrian_detection'
name = 'yolov8s_mot17_det'

# Train the model
results = model.train(
    data="configs/mot17.yaml",
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    project=project_name,
    name=name
)

best_model_path = os.path.join("runs", "detect", project_name, name, "weights", "best.pt")
save_path = os.path.join("models", "detector", "yolov8s_mot17.pt")

os.makedirs("models/detector", exist_ok=True)
shutil.copy(best_model_path, save_path)

print(f"Best model saved to: {save_path}")