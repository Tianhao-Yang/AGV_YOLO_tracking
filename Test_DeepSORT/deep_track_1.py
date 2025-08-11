import pyzed.sl as sl
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np


#np.float = np.float64


# === Global variables for click and selected object ===
clicked_position = None
selected_track_id = None

# === Mouse click callback ===
def mouse_callback(event, x, y, flags, param):
    global clicked_position, selected_track_id
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_position = (x, y)
        selected_track_id = None  # Reset on new click
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_position = None
        selected_track_id = None

# === Load YOLOv8 model and DeepSORT tracker ===
model = YOLO('yolov8n.pt')

tracker = DeepSort(
    max_age=60,               # Allow up to 60 frames without detection before deleting track
    n_init=2,                 # Confirm a new track after 2 consecutive detections
    max_cosine_distance=0.4   # Appearance similarity threshold (0 = identical, 1 = totally different)
)


# === Initialize ZED camera ===
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 15
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)
runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth = sl.Mat()

cv2.namedWindow("ZED + YOLOv8 + DeepSORT")
cv2.setMouseCallback("ZED + YOLOv8 + DeepSORT", mouse_callback)

print("Running ZED + YOLOv8 + DeepSORT (click to select, right-click to clear, 'q' to quit)...")

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        frame = image_zed.get_data()
        ########################## ===Debug: Convert BGRA → RGB (now frame_rgb exists)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        results = model(frame_rgb, conf=0.5, iou=0.4)[0]# conf: higher the more precise / iou: higher the more duplicate 

        detections = []

        for det in results.boxes:
              x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
              conf = float(det.conf[0])
              if conf < 0.4:  # change high for strict detection and low for tolerance detection
                  continue
              cls_id = int(det.cls[0])
              detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame_rgb)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Get bounding box: (left, top, right, bottom)
            l, t, r, b = map(int, track.to_ltrb())

            # Compute width, height, and centroid
            w, h = r - l, b - t
            cx, cy = l + w // 2, t + h // 2

            # Print centroid of tracked object
            print(f"[DeepSORT] Track ID:{track_id} centroid=({cx}, {cy})")

    # (rest of your code continues...)


            # Before calling get_value, validate cx and cy are within image bounds
            height, width = depth.get_height(), depth.get_width()

            if 0 <= cx < width and 0 <= cy < height:
                depth_val = depth.get_value(cx, cy)[1]
                if np.isnan(depth_val) or depth_val <= 0:
                    continue
            else:
                continue  # Skip invalid coordinates


            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val) or depth_val <= 0:
                continue

            cls_name = model.names[track.det_class] if track.det_class < len(model.names) else "obj"

            # Check for selection
            if clicked_position and l <= clicked_position[0] <= l + w and t <= clicked_position[1] <= t + h:
                selected_track_id = track_id
                clicked_position = None

            # Draw box and label
            color = (0, 0, 255) if track_id == selected_track_id else (0, 255, 0)
            cv2.rectangle(frame_rgb, (l, t), (l + w, t + h), color, 2)
            cv2.putText(frame_rgb, f"{cls_name} ID:{track_id} {depth_val:.0f}mm",
                        (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Optional: print info
            if track_id == selected_track_id:
                print(f"[SELECTED] {cls_name} ID:{track_id} at ({cx}, {cy}) = {depth_val:.1f} mm")

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("ZED + YOLOv8 + DeepSORT", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
