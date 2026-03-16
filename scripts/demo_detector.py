import cv2
import os
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, src_img):

        results = self.model.predict(src_img)[0]

        img = results.plot()

        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/result.jpg", img)

        return results


model_path = "models/detector/yolov8s_mot17.pt"
test_img_path = "https://cdn.theatlantic.com/media/mt/food/RTR2LP34edit.jpg"

detector = Detector(model_path)
results = detector.detect(test_img_path)