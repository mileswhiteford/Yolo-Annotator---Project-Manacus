import os
import shutil
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Install dependencies if running outside a pre-configured environment
# !pip install -q ultralytics

class YOLOInferenceRunner:
    """
    Handles running YOLO inference on a local video file and saving
    detections as individual YOLO-format .txt files (normalized coordinates)
    for use in an annotation tool.
    """

    def __init__(self, yolo_weights_path: str):
        """Initializes the YOLO model."""
        yolo_weights_path = os.path.abspath(yolo_weights_path)
        if not os.path.exists(yolo_weights_path):
            raise FileNotFoundError(f"YOLO weights not found at: {yolo_weights_path}")
        print(f"Loading YOLO model from: {yolo_weights_path}")
        try:
            # Load the model once during initialization
            self.model = YOLO(yolo_weights_path)
        except Exception as e:
            raise ValueError(f"Failed to load YOLO weights: {e}")
        
        self.yolo_weights_path = yolo_weights_path

    def run_inference_and_save(
        self,
        video_path: str,
        output_dir: str
    ) -> str:
        """
        Runs YOLO inference on a video and saves detections in the
        required annotation tool format (individual normalized .txt files).

        Args:
            video_path: Path to the local MP4 video file.
            output_dir: Directory where the output YOLO .txt files will be saved.

        Returns:
            The path to the output directory (`yolo_dir`).
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 1. Prepare Output Directory
        yolo_dir = os.path.join(output_dir, "yolo_labels")
        if os.path.exists(yolo_dir):
            shutil.rmtree(yolo_dir)
        os.makedirs(yolo_dir, exist_ok=True)
        print(f"Saving normalized YOLO output files to: {yolo_dir}")

        # 2. Get Video Dimensions for Normalization
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_width == 0 or frame_height == 0:
            raise ValueError(f"Could not read video dimensions from {video_path}")

        # 3. Run YOLO Tracking and Save Per-Frame
        frame_idx = 0
        total_detections = 0
        
        # Use track() with persist=True for object tracking across frames
        # Stream=True is efficient for video processing
        for result in self.model.track(video_path, stream=True, persist=True):
            boxes = result.boxes
            yolo_lines = []
            
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                # Extract tracking IDs (object instance IDs)
                ids = boxes.id
                if ids is not None:
                    ids = ids.cpu().numpy()
                else:
                    # If tracking fails, assign sequential IDs per frame
                    ids = np.arange(len(xyxy))
                
                # Iterate through all detected boxes in this frame
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    class_id = int(cls[i])
                    object_id = int(ids[i]) if ids is not None else i
                    
                    # Calculate Normalized YOLO format (x_center, y_center, width, height)
                    w_pixel = x2 - x1
                    h_pixel = y2 - y1
                    xc_pixel = x1 + w_pixel / 2
                    yc_pixel = y1 + h_pixel / 2
                    
                    # Normalize coordinates
                    xc_norm = xc_pixel / frame_width
                    yc_norm = yc_pixel / frame_height
                    w_norm = w_pixel / frame_width
                    h_norm = h_pixel / frame_height
                    
                    # Create the YOLO output line (object_id class_id x_c y_c w h)
                    line = f"{object_id} {class_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                    yolo_lines.append(line)
                    total_detections += 1

            # 4. Save Detections to Frame-Specific .txt File
            # The format requires zero-padding (e.g., 00000.txt, 00001.txt)
            output_filename = os.path.join(yolo_dir, f"{frame_idx:05d}.txt")

            with open(output_filename, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            frame_idx += 1
            
        print(f"\n--- Inference Complete ---")
        print(f"Frames Processed: {frame_idx}")
        print(f"Total Detections Saved: {total_detections}")
        
        # Return the directory path where the .txt files are stored
        return yolo_dir
