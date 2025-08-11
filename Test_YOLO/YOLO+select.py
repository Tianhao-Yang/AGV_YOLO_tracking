import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import numpy as np

# === Global variables for click and selected object ===
clicked_position = None
selected_box_index = None

# === Mouse click callback ===
def mouse_callback(event, x, y, flags, param):
    global clicked_position
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_position = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_position = None

# === Load YOLOv8 model ===
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

print("Running ZED + YOLOv8 (click to select a box, right-click to clear, 'q' to quit)...")

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image_zed.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Run YOLOv8 detection
        results = model(frame_rgb)[0]
        boxes = results.boxes
        selected_box_index = None

        # Identify clicked box
        if clicked_position and len(boxes) > 0:
            min_dist = float('inf')
            for i, det in enumerate(boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                if x1 <= clicked_position[0] <= x2 and y1 <= clicked_position[1] <= y2:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    dist = np.hypot(cx - clicked_position[0], cy - clicked_position[1])
                    if dist < min_dist:
                        selected_box_index = i
                        min_dist = dist

        # Draw boxes
        for i, det in enumerate(boxes):
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            depth_value = depth.get_value(cx, cy)[1]
            if np.isnan(depth_value) or depth_value <= 0:
                continue

            cls_id = int(det.cls[0])
            label = model.names[cls_id]

            if i == selected_box_index:
                print(f"[SELECTED] {label} at ({cx}, {cy}) = {depth_value:.1f} mm")
                color = (0, 0, 255)  # Red for selected
            else:
                color = (0, 255, 0)  # Green for others

            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_rgb, f"{label} {depth_value:.0f}mm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("ZED + YOLOv8", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
