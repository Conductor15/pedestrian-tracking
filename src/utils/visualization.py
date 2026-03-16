import cv2
import numpy as np


def draw_detection(img, bboxes, scores, ids, mask_alpha=0.3):
    """
    Draw tracking results on image.

    Args:
        img (np.ndarray): Input image.
        bboxes (np.ndarray): Bounding boxes (x1, y1, x2, y2).
        scores (list): Detection confidence scores.
        ids (list): Track IDs from tracker.
        mask_alpha (float): Transparency for mask overlay.

    Returns:
        np.ndarray: Image with drawn detections.
    """
    height, width = img.shape[:2]
    np.random.seed(0)
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(100, 3))
    
    mask_img = img.copy()
    det_img = img.copy()
    
    size = min([height, width]) * 0.0006
    text_thickness = int(min([height, width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for bbox, score, id_ in zip(bboxes, scores, ids):
        color = colors[id_ % 100]
    
        x1, y1, x2, y2 = bbox.astype(int)
    
        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
    
        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
    
        caption = f'ID {id_} {int(score*100)}%'
    
        (tw, th), _ = cv2.getTextSize(
            text=caption,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness
        )
    
        th = int(th * 1.2)
    
        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
    
        cv2.putText(
            det_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )
    
        cv2.putText(
            mask_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )
    
    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)