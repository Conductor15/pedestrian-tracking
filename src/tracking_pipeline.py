import os
import cv2
import numpy as np

from src.utils.visualization import draw_detection


def video_tracking(video_path, detector, tracker, is_save_result=False, save_dir="tracking_results"):
    """
    Run pedestrian tracking on a video using a detector and tracker.

    This function reads frames from a video, performs object detection,
    passes detections to a tracker to maintain object identities across frames,
    and optionally saves the tracking visualization as a new video.

    Args:
        video_path (str):
            Path to the input video.

        detector (object):
            Detector instance with a `detect(frame)` method that returns:
            (bboxes, scores, class_ids).

        tracker (object):
            Tracker instance with a `tracking()` method that assigns IDs
            to detected objects across frames.

        is_save_result (bool, optional):
            If True, save the tracking result as a video.
            Default is False.

        save_dir (str, optional):
            Directory where the output video will be saved.
            Default is "tracking_results".

    Returns:
        list[np.ndarray]:
            A list containing tracking results for each frame.
            Each element typically contains bounding boxes and track IDs
            predicted by the tracker.

    Workflow:
        1. Read frames from the input video.
        2. Detect pedestrians using the detector.
        3. Track detected objects using the tracker.
        4. Draw bounding boxes and track IDs on frames.
        5. Optionally save the output video.

    Notes:
        - Bounding boxes format: (x1, y1, x2, y2)
        - Tracking results usually contain:
              [x1, y1, x2, y2, track_id]
        - Visualization is handled by `draw_detection`.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if is_save_result:
        os.makedirs(save_dir, exist_ok=True)
    
        # Get the video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    
        # Define the codec and create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
        save_result_name = 'output_video3.avi'
        save_result_path = os.path.join(save_dir, save_result_name)

        out = cv2.VideoWriter(save_result_path, fourcc, fps, (width, height))

    all_tracking_results = []
    tracked_ids = np.array([], dtype=np.int32)
    
    while True:
        ret, frame = cap.read()
    
        if not ret:
            break
    
        detector_results = detector.detect(frame)
        bboxes, scores, class_ids = detector_results
    
        tracker_pred = tracker.tracking(
            origin_frame=frame,
            bboxes=bboxes,
            scores=scores
        )
    
    
        if tracker_pred.size > 0:
            bboxes = tracker_pred[:, :4]
            tracking_ids = tracker_pred[:, 4].astype(int)
            
            conf_scores = scores[:len(bboxes)]
        
            # Get new tracking IDs
            new_ids = np.setdiff1d(tracking_ids, tracked_ids)
        
            # Store new tracking IDs
            tracked_ids = np.concatenate((tracked_ids, new_ids))
        
            result_img = draw_detection(
                img=frame,
                bboxes=bboxes,
                scores=conf_scores,
                ids=tracking_ids
            )
        else:
            result_img = frame

        all_tracking_results.append(tracker_pred)
    
        if is_save_result == 1:
            out.write(result_img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release video capture
    cap.release()
    if is_save_result:
        out.release()
    
    cv2.destroyAllWindows()
    
    return all_tracking_results
