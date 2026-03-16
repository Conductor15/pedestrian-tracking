from ultralytics import YOLO

# Load the trained model
# model_path = os.path.join(
#     project_name, name, 'weights/best.pt'
# )
project_name = 'pedestrian_detection'
model_path = "models/detector/yolov8s_mot17.pt"

model = YOLO(model_path)

metrics = model.val(
    data="configs/mot17.yaml",
    project=project_name,
    name='detect/val'
)