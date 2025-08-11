import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import numpy as np

# === Global variables for click and selected object ===
clicked_position = None
selected_track_id = None

# === Mouse click callback ===
def mouse_callback(event, x, y, flags, param):
    global clicked_position, selected_track_id
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_position = (x, y)
        selected_track_id = None  # reset
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_position = None
        selected_track_id = None

# === Load YOLOv8 model with tracking support ===
model = YOLO('yolov8n.pt')

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

# Open display window and set callback
cv2.namedWindow("ZED + YOLOv8")
cv2.setMouseCallback("ZED + YOLOv8", mouse_callback)

print("Running ZED + YOLOv8 with Multi-Object Tracking...")

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        frame = image_zed.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Run YOLOv8 detection + tracking
        results = model.track(frame_rgb, persist=True, conf=0.4)[0]  # conf threshold optional
        boxes = results.boxes

        # === Determine clicked target's track ID ===
        if clicked_position and len(boxes) > 0 and selected_track_id is None:
            min_dist = float('inf')
            for i, det in enumerate(boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                if x1 <= clicked_position[0] <= x2 and y1 <= clicked_position[1] <= y2:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    dist = np.hypot(cx - clicked_position[0], cy - clicked_position[1])
                    if dist < min_dist:
                        min_dist = dist
                        selected_track_id = int(det.id[0]) if det.id is not None else -1

        # === Draw all detections with ID ===
        for det in boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_value = depth.get_value(cx, cy)[1]

            if np.isnan(depth_value) or depth_value <= 0:
                continue

            cls_id = int(det.cls[0])
            label = model.names[cls_id]
            track_id = int(det.id[0]) if det.id is not None else -1

            if selected_track_id is not None and track_id == selected_track_id:
                print(f"[SELECTED] {label} ID:{track_id} at ({cx}, {cy}) = {depth_value:.1f} mm")
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green

            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_rgb, f"{label} ID:{track_id} {depth_value:.0f}mm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("ZED + YOLOv8", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
