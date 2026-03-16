from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, src_img):
        results = self.model(src_img)[0]

        bboxes = results.boxes.xywh.cpu().numpy()
        bboxes[:, :2] = bboxes[:, :2] - (bboxes[:, 2:] / 2)

        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        return bboxes, scores, class_ids