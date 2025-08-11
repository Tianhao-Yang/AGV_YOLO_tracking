import pyzed.sl as sl
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import time
import threading
import subprocess
import re
import psutil

# Global variable to share stats
system_stats = {"cpu": "N/A", "gpu": "N/A", "ram": "N/A"}
# monitor CPU and GPU
def monitor_system(interval=1):
    def _monitor():
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while True:
            line = process.stdout.readline().decode('utf-8')
            if not line:
                break

            match = re.search(r'RAM (\d+)/(\d+)MB.*GR3D_FREQ (\d+)%', line)
            if match:
                used_ram, total_ram, gpu = match.groups()
                cpu = psutil.cpu_percent(interval=None)
                system_stats["cpu"] = f"{cpu:.1f}%"
                system_stats["gpu"] = f"{gpu}%"
                system_stats["ram"] = f"{used_ram}/{total_ram} MB"

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()

# === Check CUDA
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# === Load YOLOv8 model ===
model = YOLO('yolov8n.pt')
model.to("cpu")  # Explicitly move to GPU  ("cuda" or "cpu")

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

monitor_system()  # Start monitoring CPU and GPU 
time.sleep(3)  # wait briefly before main loop


while True:
    t0 = time.time()

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        continue
    t1 = time.time()

    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

    frame = image_zed.get_data()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    t2 = time.time()

    results = model(frame_rgb)[0]
    t3 = time.time()

    for det in results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        depth_value = depth.get_value(cx, cy)[1]
        if np.isnan(depth_value) or depth_value <= 0:
            continue

        cls_id = int(det.cls[0])
        label = model.names[cls_id]

        #print(f"{label} at ({cx}, {cy}) = {depth_value:.1f} mm")

        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame_rgb, f"{label} {depth_value:.0f}mm", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    t4 = time.time()

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("ZED + YOLOv8", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t5 = time.time()

    results = model(frame_rgb,verbose = False)[0]#suppress YOLO output 





    # own output
    h,w = frame_rgb.shape[:2]
    print(f"image size: {h}*{w}") 

    print(f"Selected Target:")
    print(f"{label} at ({cx}, {cy}) = {depth_value:.1f} mm")

    print(f"Processing time:")
    print(f"Times [ms] -> grab: {(t1-t0)*1000:.1f}\n preprocess: {(t2-t1)*1000:.1f}\n "
          f"inference: {(t3-t2)*1000:.1f}\n postprocess: {(t4-t3)*1000:.1f}\n display: {(t5-t4)*1000:.1f}\n "
          f"total: {(t5-t0)*1000:.1f}")
    if system_stats["cpu"] != "N/A":
        print(f"System Stats: CPU: {system_stats['cpu']}, GPU: {system_stats['gpu']}, RAM: {system_stats['ram']}")
    else:
        print("System Stats: initializing...")

    print()
zed.close()
cv2.destroyAllWindows()
