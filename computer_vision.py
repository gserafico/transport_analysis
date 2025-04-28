"""
Roboflow Traffic Analysis Script

This script utilizes a specified Roboflow model and the Supervision library to
perform object detection, tracking, and line-crossing counts on traffic video
footage. It aggregates counts per allowed class over defined time intervals and
saves the results to CSV files. Optional visualization is included.

Setup:
1.  Install required libraries: `pip install -r requirements.txt`
2.  Set the `ROBOFLOW_API_KEY` environment variable or replace the placeholder
    in the script (strongly recommended to use environment variables).
3.  Configure `ROBOFLOW_PROJECT_ID` and `ROBOFLOW_VERSION_NUMBER`.
4.  Update `ALLOWED_CLASSES` with the specific object classes you want to count.
    Ensure these names match the class names in your Roboflow model.
5.  Place video files in the `INPUT_FOLDER`.
6.  The script will create the `OUTPUT_FOLDER` if it doesn't exist.
7.  Adjust other configuration parameters (thresholds, line position, etc.).
"""

import contextlib
import glob
import logging
import math
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from roboflow import Roboflow
from tqdm import tqdm

# --- Configuration ---

# 1. Roboflow Model Configuration
ROBOFLOW_API_KEY: Optional[str] = os.getenv("ROBOFLOW_API_KEY", "b0TYZPLasejXmHivS47F") #Check roboflow for different models
ROBOFLOW_PROJECT_ID: str = "car-detection-model-bwjpb"
ROBOFLOW_VERSION_NUMBER: int = 8

# 2. Class Filtering: Specify object classes to detect, track, and count.
#    Case-insensitive matching will be used against the model's class names.
ALLOWED_CLASSES: List[str] = [
    'Bus', 'Jeepney', 'Motorcycle', 'SUV', 'Sedan', 'Truck', 'Van'
]

# 3. Visualization Settings
ENABLE_VISUALIZATION: bool = True
VISUALIZE_ANCHOR_POINTS: bool = True

# 4. Input/Output Folders
INPUT_FOLDER: str = "traffic_footage"
OUTPUT_FOLDER: str = "traffic_analysis_output"

# 5. Analysis Parameters
CONFIDENCE_THRESHOLD_PERCENT: int = 30 # Roboflow predict() expects 0-100
OVERLAP_THRESHOLD_PERCENT: int = 40    # Roboflow predict() expects 0-100
FRAME_SKIP: int = 1
TIME_INTERVAL_SECONDS: int = 60

# 6. Line Counter Configuration (Normalized Coordinates 0.0 to 1.0)
LINE_START_NORM: sv.Point = sv.Point(x=0.1, y=0.5)
LINE_END_NORM: sv.Point = sv.Point(x=0.9, y=0.5)
LINE_TRIGGER_ANCHOR: sv.Position = sv.Position.CENTER

# 7. Optional: Limit Video Processing Duration
MAX_DURATION_SECONDS: Optional[float] = None #ex. 300 for first 5 mins only

# --- Derived Constants & Setup ---
OUTPUT_CSV_SUFFIX: str = "vehicle_count.csv"
VISUALIZATION_WINDOW_NAME: str = "Traffic Analysis - Press 'q' to Quit"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --- Helper Functions ---

def ensure_dir_exists(dir_path: str) -> bool:
    """
    Creates a directory if it doesn't already exist.

    Args:
        dir_path: The path to the directory to create.

    Returns:
        True if the directory exists or was created successfully, False otherwise.
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
            return True
        except OSError as e:
            logging.error(f"Failed to create directory {dir_path}: {e}")
            return False
    return True

def get_video_files(folder_path: str) -> List[str]:
    """
    Finds video files with common extensions in the specified folder.

    Args:
        folder_path: The path to the directory containing video files.

    Returns:
        A list of paths to the found video files. Returns an empty list if
        the folder doesn't exist or no videos are found.
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    logging.info(f"Searching for video files in '{folder_path}'")
    if not os.path.isdir(folder_path):
        logging.error(f"Input folder not found: {folder_path}")
        return []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return video_files

def initialize_roboflow_model(
    api_key: Optional[str], project_id: str, version_number: int
    ) -> Tuple[Optional[Any], Dict[int, str]]:
    """
    Initializes the Roboflow model via API and retrieves the class map.

    Includes a hardcoded fallback map specific to the project ID if automatic
    retrieval fails.

    Args:
        api_key: The Roboflow API key.
        project_id: The Roboflow project ID (workspace URL slug).
        version_number: The specific model version number.

    Returns:
        A tuple containing:
        - The initialized Roboflow model object (or None on failure).
        - A dictionary mapping class IDs (int) to class names (str)
          (or an empty dictionary on failure).
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE": # Check placeholder
        logging.error("Roboflow API Key is not set or is using the placeholder value.")
        logging.error("Set the ROBOFLOW_API_KEY environment variable or update the script.")
        return None, {}

    model: Optional[Any] = None
    class_map: Dict[int, str] = {}
    # Fallback map specific to project "car-detection-model-bwjpb", version 8 classes
    manual_class_map: Dict[int, str] = {
        0: 'Bicycle', 1: 'Bus', 2: 'Jeepney', 3: 'Motorcycle', 4: 'Multicab',
        5: 'SUV', 6: 'Sedan', 7: 'Tricycle', 8: 'Truck', 9: 'Van'
    }

    try:
        logging.info("Initializing Roboflow client")
        rf = Roboflow(api_key=api_key)
        project = rf.project(project_id)
        version = project.version(version_number)
        model = version.model

        if hasattr(version, 'classes') and version.classes:
            raw_map = version.classes
            first_key = next(iter(raw_map.keys()), None)
            if isinstance(first_key, str): # {name: id} format
                class_map = {int(v): k for k, v in raw_map.items() if isinstance(v, (int, float))}
            elif isinstance(first_key, int): # {id: name} format
                class_map = {k: str(v) for k, v in raw_map.items() if isinstance(v, str)}
            logging.info(f"Model Class Map (ID: Name) retrieved from Roboflow: {class_map}")
        else:
            logging.warning("Could not automatically retrieve class names from Roboflow version.")

        if not class_map:
            logging.warning("Using manually defined class map as fallback.")
            class_map = manual_class_map
            logging.info(f"Manual Class Map (ID: Name): {class_map}")

        if model is None: raise ValueError("Failed to load model from Roboflow version.")
        if not class_map: raise ValueError("Class map is empty after retrieval and fallback.")

        logging.info("Roboflow model initialized successfully.")
        return model, class_map

    except Exception as e:
        logging.error(f"CRITICAL ERROR during Roboflow Initialization: {e}", exc_info=True)
        return None, {}

def setup_video_processing(
    video_path: str
    ) -> Tuple[Optional[cv2.VideoCapture], float, int, int, float]:
    """
    Opens a video file using OpenCV and retrieves its properties.

    Args:
        video_path: The path to the video file.

    Returns:
        A tuple containing:
        - OpenCV VideoCapture object (or None if opening failed).
        - Frames per second (float, defaults to 30.0 if detection fails).
        - Frame width (int).
        - Frame height (int).
        - Total video duration in seconds (float).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return None, 0.0, 0, 0, 0.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        logging.warning(f"Invalid FPS ({fps}) detected. Assuming 30 FPS.")
        fps = 30.0
    video_duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

    logging.info(f"Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS, "
                 f"{total_frames} Frames ({video_duration_sec:.2f}s)")
    return cap, fps, frame_width, frame_height, video_duration_sec

def setup_annotators(visualize: bool, visualize_anchors: bool) -> Dict[str, Optional[Any]]:
    """
    Initializes Supervision annotator objects based on visualization settings.

    Args:
        visualize: If True, enable visualization annotators.
        visualize_anchors: If True (and visualize is True), enable anchor point dots.

    Returns:
        A dictionary containing initialized annotator objects (or None).
        Keys: 'box', 'label', 'line', 'dot'.
    """
    annotators: Dict[str, Optional[Any]] = {
        'box': None, 'label': None, 'line': None, 'dot': None
    }
    if visualize:
        annotators['box'] = sv.BoxAnnotator(thickness=1)
        annotators['label'] = sv.LabelAnnotator(
            text_position=sv.Position.BOTTOM_CENTER, text_scale=0.3,
            text_thickness=1, text_padding=1
        )
        #Line annotator, useful to get live counts, commented out for better visibility for the prototype. 
        #annotators['line'] = sv.LineZoneAnnotator( 
        #    thickness=2, text_thickness=1, text_scale=0.5,
        #    text_color=sv.Color.BLACK, color=sv.Color.WHITE
        #)
        if visualize_anchors:
            annotators['dot'] = sv.DotAnnotator(color=sv.Color.RED, radius=3)
        logging.info(f"Visualization Enabled. Press 'q' in window '{VISUALIZATION_WINDOW_NAME}' to Quit.")
    return annotators

def parse_roboflow_predictions(
    predictions_json: Dict[str, Any],
    allowed_id_to_name_map: Dict[int, str]
    ) -> sv.Detections:
    """
    Parses the JSON output from Roboflow model.predict() into Supervision
    Detections object, filtering only for classes specified in the
    allowed_id_to_name_map.

    Args:
        predictions_json: The raw JSON dictionary from model.predict().json().
        allowed_id_to_name_map: A dictionary mapping allowed class IDs (int)
                                to their names (str).

    Returns:
        A sv.Detections object containing filtered detections. Returns
        sv.Detections.empty() if no allowed objects are found.
    """
    xyxy_list, confidence_list, class_id_list = [], [], []

    if 'predictions' in predictions_json and isinstance(predictions_json['predictions'], list):
        for pred in predictions_json['predictions']:
            class_id_raw = pred.get('class_id')
            class_id = int(class_id_raw) if class_id_raw is not None else None

            # Filter based on the pre-filtered map of allowed IDs
            if class_id is not None and class_id in allowed_id_to_name_map:
                x_center = pred.get('x')
                y_center = pred.get('y')
                width = pred.get('width')
                height = pred.get('height')
                confidence = pred.get('confidence')

                if all(v is not None for v in [x_center, y_center, width, height, confidence]):
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    xyxy_list.append([x_min, y_min, x_max, y_max])
                    confidence_list.append(confidence)
                    class_id_list.append(class_id)

    if xyxy_list:
        return sv.Detections(
            xyxy=np.array(xyxy_list),
            confidence=np.array(confidence_list),
            class_id=np.array(class_id_list).astype(int)
        )
    else:
        return sv.Detections.empty()

def format_results_to_dataframe(
    interval_class_counts: Dict[int, Dict[str, Dict[str, int]]],
    time_interval_sec: int,
    all_counted_class_names: List[str]
    ) -> Optional[pd.DataFrame]:
    """
    Formats the aggregated count data into a pandas DataFrame.

    Includes columns for each counted class (In/Out) and totals per interval.

    Args:
        interval_class_counts: The nested dictionary holding counts.
                               Format: {interval_idx: {class_name: {'in': N, 'out': M}}}
        time_interval_sec: The duration of each interval in seconds.
        all_counted_class_names: A list of class names for which counts
                                 should be included as columns.

    Returns:
        A pandas DataFrame with the formatted results, or None if input is empty.
    """
    if not interval_class_counts:
        logging.info("No line crossings were detected; DataFrame will be empty.")
        return None # Return None if no data to process

    data_rows = []
    # Ensure all intervals from 0 up to the max index found are represented
    last_interval_idx = max(interval_class_counts.keys()) if interval_class_counts else -1

    for interval_index in range(last_interval_idx + 1):
        start_time = interval_index * time_interval_sec
        end_time = (interval_index + 1) * time_interval_sec
        row: Dict[str, Any] = {
            'Interval': interval_index,
            'Start Time (s)': f"{start_time:.2f}",
            'End Time (s)': f"{end_time:.2f}"
        }
        interval_total_in, interval_total_out = 0, 0

        for class_name in all_counted_class_names:
            # Get counts for this class in this interval, default to 0 if not present
            class_counts = interval_class_counts.get(interval_index, {}).get(class_name, {'in': 0, 'out': 0})
            in_count = class_counts['in']
            out_count = class_counts['out']
            row[f'{class_name}_In'] = in_count
            row[f'{class_name}_Out'] = out_count
            interval_total_in += in_count
            interval_total_out += out_count

        row['Total_In'] = interval_total_in
        row['Total_Out'] = interval_total_out
        row['Total_Crossing'] = interval_total_in + interval_total_out
        data_rows.append(row)

    # Define final column order
    base_columns = ['Interval', 'Start Time (s)', 'End Time (s)']
    class_columns = []
    for class_name in sorted(all_counted_class_names): # Sort class columns alphabetically
        class_columns.append(f'{class_name}_In')
        class_columns.append(f'{class_name}_Out')
    total_columns = ['Total_In', 'Total_Out', 'Total_Crossing']
    all_columns = base_columns + class_columns + total_columns

    results_df = pd.DataFrame(data_rows, columns=all_columns)
    results_df.fillna(0, inplace=True) # Should not be necessary but safe

    # Convert count columns to integer type robustly
    count_cols = [col for col in results_df.columns if '_In' in col or '_Out' in col]
    for col in count_cols:
         results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)

    return results_df

# --- Main Processing Function ---

def analyze_traffic_video(
    video_path: str,
    model: Any,
    class_id_to_name_map: Dict[int, str],
    allowed_class_names: List[str],
    frame_skip: int,
    confidence_thresh_percent: int,
    overlap_thresh_percent: int,
    time_interval_sec: int,
    visualize: bool,
    visualize_anchors: bool,
    max_duration_sec: Optional[float]
    ) -> Optional[pd.DataFrame]:
    """
    Analyzes a video file using a Roboflow model, tracks objects, and counts
    line crossings per class per time interval.

    Args:
        video_path: Path to the input video file.
        model: Initialized Roboflow model object.
        class_id_to_name_map: Mapping from class ID (int) to class name (str)
                              provided by the Roboflow model.
        allowed_class_names: List of class names (strings) to be tracked/counted.
        frame_skip: Interval for processing frames.
        confidence_thresh_percent: Detection confidence threshold (0-100).
        overlap_thresh_percent: Detection overlap threshold for NMS (0-100).
        time_interval_sec: Duration of each aggregation interval (seconds).
        visualize: Whether to display annotated video during processing.
        visualize_anchors: Whether to draw anchor points on tracked objects.
        max_duration_sec: Optional maximum duration of video to process (seconds).

    Returns:
        A pandas DataFrame containing aggregated counts per class per interval,
        or None if processing fails or is interrupted.
    """
    cap: Optional[cv2.VideoCapture] = None
    processed_frame_count: int = 0
    total_inference_time: float = 0.0 # Track only inference time
    visualization_active: bool = visualize
    printed_debug_info: bool = False

    # Filter class map based on allowed names provided
    allowed_id_to_name_map: Dict[int, str] = {
        cid: name for cid, name in class_id_to_name_map.items()
        if name in allowed_class_names
    }
    if not allowed_id_to_name_map:
        logging.error("No allowed classes found in the model's map. Cannot proceed.")
        return None
    logging.info(f"Counting crossings for classes/IDs: {allowed_id_to_name_map}")

    results_df: Optional[pd.DataFrame] = None
    try:
        cap, fps, width, height, _ = setup_video_processing(video_path)
        if cap is None: return None

        line_start_pixels = sv.Point(int(LINE_START_NORM.x * width), int(LINE_START_NORM.y * height))
        line_end_pixels = sv.Point(int(LINE_END_NORM.x * width), int(LINE_END_NORM.y * height))
        line_zone = sv.LineZone(start=line_start_pixels, end=line_end_pixels, triggering_anchors=[LINE_TRIGGER_ANCHOR])
        tracker = sv.ByteTrack()
        logging.info(f"Line Zone: Start={line_start_pixels}, End={line_end_pixels}, Anchor={LINE_TRIGGER_ANCHOR}")

        annotators = setup_annotators(visualize, visualize_anchors)

        interval_class_counts: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {'in': 0, 'out': 0}))
        crossed_this_interval: Set[Tuple[Optional[int], str]] = set()
        current_interval_index: int = -1

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = total_frames
        desc = f"Analyzing {os.path.basename(video_path)}"
        if max_duration_sec is not None:
            limit_frame = int(max_duration_sec * fps)
            if total_frames > 0: frames_to_process = min(total_frames, limit_frame)
            else: frames_to_process = limit_frame
            desc += f" (max {max_duration_sec:.0f}s)"
        logging.info(f"Approx. frames to process: {frames_to_process}")

        with tqdm(total=frames_to_process, desc=desc, unit="frame", leave=True) as pbar:
            while True:
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_time_sec = frame_num / fps if fps > 0 else 0.0

                if max_duration_sec is not None and current_time_sec >= max_duration_sec:
                    logging.info(f"Reached time limit ({max_duration_sec:.1f}s).")
                    if pbar.n < pbar.total: pbar.n = pbar.total; pbar.refresh()
                    break

                ret, frame = cap.read()
                if not ret:
                    logging.info("End of video file reached.")
                    if pbar.n < pbar.total: pbar.n = pbar.total; pbar.refresh()
                    break

                if pbar.n < pbar.total: pbar.update(1)
                if frame_num % frame_skip != 0: continue

                new_interval_index = math.floor(current_time_sec / time_interval_sec)
                if new_interval_index > current_interval_index:
                    crossed_this_interval = set()
                    current_interval_index = new_interval_index

                frame_copy_for_vis = frame.copy() if visualization_active else None
                detections = sv.Detections.empty()
                try:
                    infer_start = time.time()
                    results_json = model.predict(frame, confidence=confidence_thresh_percent, overlap=overlap_thresh_percent).json()
                    infer_end = time.time()
                    total_inference_time += (infer_end - infer_start)
                    processed_frame_count += 1

                    detections = parse_roboflow_predictions(results_json, allowed_id_to_name_map)

                    if not printed_debug_info and results_json.get('predictions'):
                         logging.info("--- Sample Raw Prediction Data ---")
                         for i, p in enumerate(results_json['predictions'][:5]): logging.info(f"  Pred {i}: {p}")
                         logging.info("--- End Sample ---")
                         printed_debug_info = True

                except Exception as infer_e:
                    logging.error(f"Error during prediction on frame {frame_num}: {infer_e}")
                    continue

                if not detections.is_empty():
                    try:
                        detections = tracker.update_with_detections(detections=detections)
                    except Exception as track_e:
                         logging.error(f"Error updating tracker on frame {frame_num}: {track_e}")
                         continue

                try:
                    crossing_in_mask, crossing_out_mask = line_zone.trigger(detections=detections)
                    if np.any(crossing_in_mask):
                        dets_in = detections[crossing_in_mask]
                        for i in range(len(dets_in)):
                            tracker_id = dets_in.tracker_id[i] if dets_in.tracker_id is not None and i < len(dets_in.tracker_id) else None
                            class_id = dets_in.class_id[i]
                            if tracker_id is not None and (tracker_id, 'in') not in crossed_this_interval:
                                class_name = allowed_id_to_name_map.get(class_id)
                                if class_name:
                                    interval_class_counts[current_interval_index][class_name]['in'] += 1
                                    crossed_this_interval.add((tracker_id, 'in'))
                    if np.any(crossing_out_mask):
                        dets_out = detections[crossing_out_mask]
                        for i in range(len(dets_out)):
                            tracker_id = dets_out.tracker_id[i] if dets_out.tracker_id is not None and i < len(dets_out.tracker_id) else None
                            class_id = dets_out.class_id[i]
                            if tracker_id is not None and (tracker_id, 'out') not in crossed_this_interval:
                                class_name = allowed_id_to_name_map.get(class_id)
                                if class_name:
                                    interval_class_counts[current_interval_index][class_name]['out'] += 1
                                    crossed_this_interval.add((tracker_id, 'out'))
                except Exception as line_e:
                    logging.error(f"Error during line zone trigger on frame {frame_num}: {line_e}")
                    continue

                if visualization_active and frame_copy_for_vis is not None:
                    try:
                        labels = []
                        if not detections.is_empty():
                            for i in range(len(detections)):
                                tracker_id = detections.tracker_id[i] if detections.tracker_id is not None and i < len(detections.tracker_id) else None
                                class_id = detections.class_id[i]
                                conf = detections.confidence[i] if detections.confidence is not None and i < len(detections.confidence) else 0.0
                                id_part = f"ID:{tracker_id}" if tracker_id is not None else "Det"
                                class_name = allowed_id_to_name_map.get(class_id, f"ClsID:{class_id}")
                                labels.append(f"{id_part} {class_name} {conf:.2f}")

                        annotated_frame = frame_copy_for_vis
                        if annotators['box']: annotated_frame = annotators['box'].annotate(scene=annotated_frame, detections=detections)
                        if annotators['label']: annotated_frame = annotators['label'].annotate(scene=annotated_frame, detections=detections, labels=labels)
                        if annotators['line']: annotated_frame = annotators['line'].annotate(frame=annotated_frame, line_counter=line_zone)
                        if annotators['dot'] and not detections.is_empty(): annotated_frame = annotators['dot'].annotate(scene=annotated_frame, detections=detections)

                        cv2.imshow(VISUALIZATION_WINDOW_NAME, annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logging.info("'q' pressed. Stopping.")
                            visualization_active = False; break

                        with contextlib.suppress(cv2.error):
                             if cv2.getWindowProperty(VISUALIZATION_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                                 logging.info("Visualization window closed.")
                                 visualization_active = False; break
                    except Exception as vis_e:
                        logging.error(f"Visualization error frame {frame_num}: {vis_e}", exc_info=False)
                        visualization_active = False # Stop visualizing on error

        results_df = format_results_to_dataframe(
            interval_class_counts, time_interval_sec, sorted(list(allowed_id_to_name_map.values()))
        )
        avg_fps = processed_frame_count / total_inference_time if total_inference_time > 0 else 0
        logging.info(f"Finished processing. Analyzed {processed_frame_count} frames. Avg Inference FPS: {avg_fps:.2f}")

    except Exception as e:
        logging.error(f"Unexpected error during video processing for {video_path}: {e}", exc_info=True)
        results_df = None
    finally:
        if cap is not None: cap.release(); logging.debug("Video capture released.")
        if visualize:
            with contextlib.suppress(Exception): cv2.destroyAllWindows(); cv2.waitKey(1)
            logging.debug("Visualization windows closed.")

    return results_df

# --- Main Execution Function ---

def main():
    """Main function to orchestrate the traffic analysis process."""
    print("--- Vehicle Counter - CCTV Traffic Analysis ---")
    logging.info(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not ensure_dir_exists(OUTPUT_FOLDER): return
    if not os.path.isdir(INPUT_FOLDER): logging.error(f"Input folder '{INPUT_FOLDER}' not found."); return

    model, class_map = initialize_roboflow_model(ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_ID, ROBOFLOW_VERSION_NUMBER)
    if model is None or not class_map: logging.critical("Model init failed. Exiting."); return

    model_class_names = set(class_map.values())
    allowed_names_in_model: List[str] = []
    missing_classes: List[str] = []
    for req_name in ALLOWED_CLASSES:
        found = False
        for model_name in model_class_names:
            if model_name.lower() == req_name.lower(): allowed_names_in_model.append(model_name); found = True; break
        if not found: missing_classes.append(req_name)

    if not allowed_names_in_model: logging.error(f"None of ALLOWED_CLASSES match model names. Check config."); return
    logging.info(f"Classes configured & found in model: {allowed_names_in_model}")
    if missing_classes: logging.warning(f"Requested classes NOT found (ignored): {missing_classes}")

    video_files = get_video_files(INPUT_FOLDER)
    if not video_files: logging.error(f"No video files found in '{INPUT_FOLDER}'."); return
    logging.info(f"Found {len(video_files)} video file(s) to process:")
    for i, vf in enumerate(video_files[:5]): logging.info(f" - {os.path.basename(vf)}")
    if len(video_files) > 5: logging.info(f" - (... and {len(video_files) - 5} more)")

    overall_start_time = time.time()
    successful_files, failed_files, empty_results_files = 0, 0, 0

    for video_path in video_files:
        base_filename = os.path.basename(video_path)
        output_filename_base = os.path.splitext(base_filename)[0].replace(" ", "_")
        output_csv_path = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}{OUTPUT_CSV_SUFFIX}")

        logging.info(f"\n--- Processing Video: {base_filename} ---")
        start_analysis_time = time.time()

        results_df = analyze_traffic_video(
            video_path=video_path, model=model, class_id_to_name_map=class_map,
            allowed_class_names=allowed_names_in_model, frame_skip=FRAME_SKIP,
            confidence_thresh_percent=CONFIDENCE_THRESHOLD_PERCENT,
            overlap_thresh_percent=OVERLAP_THRESHOLD_PERCENT,
            time_interval_sec=TIME_INTERVAL_SECONDS, visualize=ENABLE_VISUALIZATION,
            visualize_anchors=VISUALIZE_ANCHOR_POINTS, max_duration_sec=MAX_DURATION_SECONDS
        )
        processing_time = time.time() - start_analysis_time

        if results_df is None:
            logging.error(f"Analysis failed/interrupted for {base_filename} ({processing_time:.2f}s).")
            failed_files += 1
        elif results_df.empty or ('Total_Crossing' in results_df.columns and results_df['Total_Crossing'].sum() == 0):
             logging.info(f"Analysis complete ({processing_time:.2f}s), but no crossings detected. No CSV saved.")
             empty_results_files += 1
        else:
            try:
                results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
                logging.info(f"Analysis complete ({processing_time:.2f}s). Results saved to: {output_csv_path}")
                logging.info("Total Crossings Summary (from CSV):")
                summary_total_in = results_df['Total_In'].sum()
                summary_total_out = results_df['Total_Out'].sum()
                counted_classes_in_df = set()
                for col in results_df.columns:
                     if col.endswith('_In') and col != 'Total_In': counted_classes_in_df.add(col.replace('_In',''))
                     elif col.endswith('_Out') and col != 'Total_Out': counted_classes_in_df.add(col.replace('_Out',''))
                for class_name in sorted(list(counted_classes_in_df)):
                    in_col, out_col = f'{class_name}_In', f'{class_name}_Out'
                    total_in = results_df[in_col].sum() if in_col in results_df.columns else 0
                    total_out = results_df[out_col].sum() if out_col in results_df.columns else 0
                    if total_in > 0 or total_out > 0: logging.info(f"    - {class_name}: In={total_in}, Out={total_out}")
                logging.info(f"    - Overall: In={summary_total_in}, Out={summary_total_out}, Total={summary_total_in + summary_total_out}")
                successful_files += 1
            except Exception as e:
                logging.error(f"Error saving results for {base_filename} to CSV: {e}", exc_info=True)
                failed_files += 1

    overall_end_time = time.time()
    logging.info("\n--- Batch Processing Summary ---")
    logging.info(f"Successfully processed and saved results for: {successful_files} video(s)")
    logging.info(f"Processed but found no line crossings for:  {empty_results_files} video(s)")
    logging.info(f"Failed to process or save results for:      {failed_files} video(s)")
    total_processed_or_attempted = successful_files + empty_results_files + failed_files
    logging.info(f"Total videos attempted:                     {total_processed_or_attempted} / {len(video_files)}")
    logging.info(f"Total batch processing time: {overall_end_time - overall_start_time:.2f} seconds.")
    logging.info(f"Output files are located in: '{os.path.abspath(OUTPUT_FOLDER)}'")
    logging.info("--- Script Finished ---")


if __name__ == "__main__":
    main()