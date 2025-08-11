import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import time

# === Command-line argument ===
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model (e.g., yolov8n.pt or yolov8s.pt)')
args = parser.parse_args()

# === Load YOLOv8 model ===
model = YOLO(args.model)
print(f"[YOLO] Loaded model: {args.model}")

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

print("Running ZED + YOLOv8 (press 'q' to quit)...")

while True:
    start_time = time.time()

    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image_zed.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # YOLO detection
        inference_start = time.time()
        results = model(frame_rgb)[0]
        inference_end = time.time()

        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            depth_value = depth.get_value(cx, cy)[1]
            if np.isnan(depth_value) or depth_value <= 0:
                continue

            cls_id = int(det.cls[0])
            label = model.names[cls_id]
            print(f"{label} at ({cx}, {cy}) = {depth_value:.1f} mm")

            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame_rgb, f"{label} {depth_value:.0f}mm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("ZED + YOLOv8", frame_bgr)

        total_time = (time.time() - start_time) * 1000
        yolo_time = (inference_end - inference_start) * 1000
        print(f"[Timing] Total: {total_time:.2f} ms | YOLO: {yolo_time:.2f} ms")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
