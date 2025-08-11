import pyzed.sl as sl
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
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
        selected_track_id = None  # Reset on new click
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_position = None
        selected_track_id = None

def _get_real_coords(point_cloud, cx, cy):
    """從點雲讀取 (X, Y, Z) 座標，失敗則回傳 nan."""
    err, pt = point_cloud.get_value(cx, cy)
    if err == sl.ERROR_CODE.SUCCESS:
        return float(pt[0]), float(pt[1]), float(pt[2])
    else:
        return float('nan'), float('nan'), float('nan')

def zed_deepsort(on_update=None):
    global clicked_position, selected_track_id

    # === Load YOLOv8 model and DeepSORT tracker ===
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(
        max_age=60,
        n_init=2,
        max_cosine_distance=0.4
    )

    # === Initialize ZED camera ===
    zed = sl.Camera()
    point_cloud = sl.Mat()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return
    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth = sl.Mat()

    cv2.namedWindow("ZED + YOLOv8 + DeepSORT")
    cv2.setMouseCallback("ZED + YOLOv8 + DeepSORT", mouse_callback)

    print("Running ZED + YOLOv8 + DeepSORT (click to select, right-click to clear, 'q' to quit)...")

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue  # 略過這一 frame

        # 1. 一波抓影像、深度、點雲資料（來源：ZED SDK 使用指南）🧠
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # 提供 X, Y, Z 座標與 RGBA 色彩 :contentReference[oaicite:5]{index=5}

        frame = image_zed.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        frame_h, frame_w = frame_rgb.shape[:2]
        center_x = frame_w // 2
        # 2. 畫面中軸紅線（像素對齊 camera self-axis）
        cv2.line(frame_rgb,
                (center_x, 0),
                (center_x, frame_h - 1),
                color=(255, 0, 0),  # RGB 紅色
                thickness=1,
                lineType=cv2.LINE_8)

        # YOLO 偵測，DeepSORT 追蹤
        results = model(frame_rgb, conf=0.5, iou=0.4)[0]
        detections = [
            ([*map(int, b.xyxy[0].tolist()[:2]), int(b.xyxy[0][2] - b.xyxy[0][0]), int(b.xyxy[0][3] - b.xyxy[0][1])],
            float(b.conf[0]),
            int(b.cls[0]))
            for b in results.boxes if float(b.conf[0]) >= 0.4
        ]

        tracks = tracker.update_tracks(detections, frame=frame_rgb)

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            cx = (l + r) // 2
            cy = (t + b) // 2
            if not (0 <= cx < frame_w and 0 <= cy < frame_h):
                continue

            _, z_mm = depth.get_value(cx, cy)
            if not np.isfinite(z_mm) or z_mm <= 0:
                continue

            cls_name = model.names[track.det_class] if track.det_class < len(model.names) else "obj"

            # 使用者點擊選擇機制
            if clicked_position and l <= clicked_position[0] <= r and t <= clicked_position[1] <= b:
                selected_track_id = track.track_id
                clicked_position = None

            color = (0, 0, 255) if track.track_id == selected_track_id else (0, 255, 0)
            cv2.rectangle(frame_rgb, (l, t), (r, b), color, 2)
            cv2.putText(frame_rgb,
                        f"{cls_name} ID:{track.track_id} Z:{z_mm:.0f}mm",
                        (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            if track.track_id == selected_track_id:
                x_mm, y_mm, z_mm = _get_real_coords(point_cloud, cx, cy)
                offset_px = cx - center_x

                err3d, pt3d = point_cloud.get_value(cx, cy)
                if err3d == sl.ERROR_CODE.SUCCESS:
                    x_mm, y_mm, z_mm3 = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])
                else:
                    x_mm = y_mm = z_mm3 = float("nan")

                offset_px = cx - center_x
                offset_mm = x_mm
                dir_px = "R" if offset_px > 0 else ("L" if offset_px < 0 else "C")
                dir_mm = "Right" if offset_mm > 0 else ("Left" if offset_mm < 0 else "Center")

                overlay = (
                    f"3D:({x_mm:.1f},{y_mm:.1f},{z_mm3:.1f})mm  "
                    f"px_off={offset_px}px({dir_px})  "
                    f"mm_off={offset_mm:.1f}mm({dir_mm})"
                )
                cv2.putText(frame_rgb, overlay, (l, b + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"[SELECTED] {cls_name} ID:{track.track_id} "
                    f"XYZ=({x_mm:.1f},{y_mm:.1f},{z_mm3:.1f})mm | "
                    f"pix_offset={offset_px}px({dir_px}) | "
                    f"mm_offset={offset_mm:.1f}mm({dir_mm})"
                    )
                if on_update:
                    on_update(track.track_id, x_mm, y_mm, z_mm)

        cv2.imshow("ZED + YOLOv8 + DeepSORT", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 結束循環
    zed.close()
    cv2.destroyAllWindows()

def handle_coords(track_id, x_mm, y_mm, z_mm):
    # 在這裡觸發任何你需要的後續邏輯或控制
    print(f"外部 callback: ID={track_id}, y={y_mm:.0f}mm, z={z_mm:.0f}mm")

if __name__ == "__main__":
    # zed_deepsort()
    zed_deepsort(on_update=handle_coords)
