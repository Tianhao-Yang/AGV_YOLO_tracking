import pyzed.sl as sl
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import time
import pygame
import serial
import time
import can
import struct
import threading
import subprocess
import sys
import pexpect

# === ZED ======================================================================================================================================================
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

latest_distance = None
latest_offset = None

def handle_coords(track_id, x_mm, y_mm, z_mm):
    global latest_distance, latest_offset
    latest_distance = z_mm / 1000.0
    latest_offset = x_mm / 1000.0
    print(f"Callback: 距離={latest_distance:.2f}m 偏移={latest_offset:.2f}m")

# def handle_coords(track_id, x_mm, y_mm, z_mm):
#     # 在這裡觸發任何你需要的後續邏輯或控制
#     print(f"外部 callback: ID={track_id}, y={y_mm:.0f}mm, z={z_mm:.0f}mm")
#     measured_distance = z_mm / 1000.0   # 轉換為公尺 (假設原本是 mm)
#     measured_offset = y_mm / 1000.0
#     return measured_distance, measured_offset
# === ZED ======================================================================================================================================================

# === PID 控制器類別 ======================================================================================================================================================
class PID:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(min(output, self.output_limit), -self.output_limit)  # 限制輸出

# === 初始化參數 ===
target_distance = 1.3             # 預定距離 (公尺)
distance_threshold = 0.15          # 距離閥值 (公尺)
offset_threshold = 0.2            # 偏移閥值 (公尺)
hold_time = 1.0                   # 閥值內/外持續時間 (秒)
dt = 0.05                         # 固定 50 毫秒 (20 Hz 控制頻率)


# 兩個馬達 PID 各自設定 (可調整參數)
motor_A_pid = PID(kp=1.5, ki=0.02, kd=0.05, output_limit=20)  # 馬達 A (距離控制)
motor_B_pid = PID(kp=0.1, ki=0.00, kd=0.03, output_limit=0.2)  # 馬達 B (位置控制)

brake_state = True
last_outside_threshold_time = time.time()
last_inside_threshold_time = time.time()

# === 控制邏輯 =======================================================================================================================================================
def control_loop(measured_distance, measured_offset, dt):
    global brake_state, last_outside_threshold_time, last_inside_threshold_time

    # 偏差計算
    distance_error = -target_distance + measured_distance #倒退
    offset_error = -measured_offset  # 左負右正 → 修正方向

    # 閥值判斷與煞車控制
    # if abs(distance_error) <= distance_threshold and abs(offset_error) <= offset_threshold:
    if abs(distance_error) <= distance_threshold:
        if time.time() - last_inside_threshold_time >= hold_time:
            brake_state = True
        else:
            last_inside_threshold_time = time.time()
    else:
        if time.time() - last_outside_threshold_time >= hold_time:
            brake_state = False
        else:
            last_outside_threshold_time = time.time()
    motor_B_cmd = motor_B_pid.compute(offset_error, dt)    # 馬達 B：位置控制

    # PID 控制
    if not brake_state:
        motor_A_cmd = motor_A_pid.compute(distance_error, dt)  # 馬達 A：距離控制
        # motor_B_cmd = motor_B_pid.compute(offset_error, dt)    # 馬達 B：位置控制
    else:
        motor_A_cmd = 0
        # motor_B_cmd = 0
        

    # 煞車互斥條件
    # brake_cmd = brake_state if motor_A_cmd == 0 and motor_B_cmd == 0 else False
    brake_cmd = brake_state if motor_A_cmd == 0 else False

    return motor_A_cmd, motor_B_cmd, brake_cmd

# === PID 控制器類別 ======================================================================================================================================================

# === ODRIVE ======================================================================================================================================================

# 初始化 pygame 手柄模組
pygame.init()
pygame.joystick.init()

# 設定 UART 連接
ser = serial.Serial('/dev/ttyACM0', 9600)  # 根據實際情況調整端口名稱
time.sleep(2)  # 等待 UART 連接穩定

# 連接手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Connected to {joystick.get_name()}")

# 馬達控制參數
velocity = 0.0  # 初始速度
max_velocity = 20.0
min_velocity = -20.0
# velocity_increment = 1.0

position = 0.0  # 初始位置
max_position = 0.2 #極限50 degree
min_position = -0.2 #極限-50 degree 要記得1.5倍皮帶輪加速轉換 編碼器讀值為-75 degree
# position_increment = 0.02

led_on = False  # LED 狀態

def setup_can_interface():
    try:
        print("Setting up CAN interface...")
        subprocess.run(['ip', 'link', 'set', 'can0', 'down'], check=False)  # 忽略失敗
        subprocess.run(['ip', 'link', 'set', 'can0', 'up', 'type', 'can', 'bitrate', '250000'], check=True)
        print("CAN interface setup complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set up CAN interface: {e}")
        sys.exit(1)





# 控制 LED 的函數
def control_led(state):
    global led_on
    if state == 'on' and not led_on:
        ser.write(b'1')  # 發送開啟 LED 指令
        led_on = True
        print("LED 開啟 (煞車)")
    elif state == 'off' and led_on:
        ser.write(b'0')  # 發送關閉 LED 指令
        led_on = False
        print("LED 關閉 (解除煞車)")
    time.sleep(0.05)

# PS2 控制線程
def joystick_control():
    global velocity, position
    velocity = 0
    position = 0
    mode_active = False  # 模式是否啟動

    # ====== super loop ================================================================================================================================
    def mode_loop():
        """進入特殊模式後的持續執行邏輯"""
        print("進入模式中...")
        while True:
            pygame.event.pump()
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

            # === 模式退出條件 ===
            if buttons[1] == 1 or buttons[3] == 1:  # 按下 B鍵 或 X鍵(煞車)
                print("退出模式，回到手動控制")
                break
            
            # measured_distance, measured_offset = handle_coords(track_id, x_mm, y_mm, z_mm)
            if latest_distance is not None and latest_offset is not None:
                speed_cmd, steer_cmd, brake_cmd = control_loop(latest_distance, latest_offset, dt)


            # speed_cmd, steer_cmd, brake_cmd = control_loop(latest_distance, latest_offset, dt)
            global velocity, position
            velocity = speed_cmd
            position = steer_cmd
            if brake_cmd is True:
                velocity = 0
                control_led('on')
                # print("煞車")
            elif brake_cmd is False:
                control_led('off')
                # print("煞車解除")

            print(f"測距: {latest_distance:.2f} m | 偏移: {latest_offset:.2f} m | "
                    f"速度命令: {speed_cmd:.2f} | 轉向命令: {steer_cmd:.2f} | 煞車: {brake_cmd}")

            time.sleep(0.1)
    # ====== super loop ================================================================================================================================ 
    
    try:
        while True:
            pygame.event.pump()
            hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

            # 按鈕 3 (X鍵）(煞車)
            if buttons[3] == 1:
                velocity = 0
                control_led('on')
                print("煞車")
            elif buttons[3] == 0:
                control_led('off')
                print("煞車解除")

            # === 按鈕 1 (B鍵）模式切換 ===
            if buttons[1] == 1 and not mode_active:
                mode_active = True
                mode_loop()         # 進入模式
                mode_active = False  # 模式結束後回到手動

            # === 一般搖桿控制 (僅當不在模式時才執行) ===
            if not mode_active and joystick.get_numaxes() >= 4:
                axis_steer = -joystick.get_axis(0)
                axis_drive = joystick.get_axis(3)
                position = max(min_position, min(max_position, axis_steer * max_position))
                velocity = max(min_velocity, min(max_velocity, -axis_drive * max_velocity))

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("退出手柄控制")
    finally:
        ser.close()
        pygame.quit()



# 驅動馬達線程
def drivemotor():
    node_id = 0  # 必須與 <odrv>.axis0.config.can.node_id 匹配，默認為 0

    bus = can.interface.Bus("can0", interface="socketcan")

    # 清空 CAN RX 緩衝區，以確保沒有舊的消息
    while not (bus.recv(timeout=0) is None):
        pass

    # 將軸設置為閉環控制狀態
    bus.send(can.Message(
        arbitration_id=(node_id << 5 | 0x07),  # 0x07: Set_Axis_State
        data=struct.pack('<I', 8),  # 8: AxisState.CLOSED_LOOP_CONTROL
        is_extended_id=False
    ))

    # 通過心跳消息等待軸進入閉環控制狀態
    for msg in bus:
        if msg.arbitration_id == (node_id << 5 | 0x01):  # 0x01: Heartbeat
            error, state, result, traj_done = struct.unpack('<IBBB', bytes(msg.data[:7]))
            if state == 8:  # 8: AxisState.CLOSED_LOOP_CONTROL
                break

    try:
        while True:
            # 發送速度命令給 ODrive
            bus.send(can.Message(
                arbitration_id=(node_id << 5 | 0x0d),  # 0x0d: Set_Input_Vel
                data=struct.pack('<ff', velocity, 0.0),  # 速度和扭矩前饋
                is_extended_id=False
            ))

            # 非阻塞接收 CAN 消息
            while True:
                msg = bus.recv(timeout=0.001)  # 設置極短超時時間
                if not msg:
                    break
                if msg.arbitration_id == (node_id << 5 | 0x09):  # 0x09: Get_Encoder_Estimates
                    pos, vel = struct.unpack('<ff', bytes(msg.data))
                    print(f"vel: {vel:.3f} [turns/s]")

            # 減少發送命令的延遲
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting CAN Bus handler...")

# 轉向馬達線程
def steermotor():
    node_id = 1  # 必須與 <odrv>.axis0.config.can.node_id 匹配，默認為 0

    bus = can.interface.Bus("can0", interface="socketcan")

    # 清空 CAN RX 緩衝區，以確保沒有舊的消息
    while not (bus.recv(timeout=0) is None):
        pass

    # 將軸設置為閉環控制狀態
    bus.send(can.Message(
        arbitration_id=(node_id << 5 | 0x07),  # 0x07: Set_Axis_State
        data=struct.pack('<I', 8),  # 8: AxisState.CLOSED_LOOP_CONTROL
        is_extended_id=False
    ))

    # 通過心跳消息等待軸進入閉環控制狀態
    for msg in bus:
        if msg.arbitration_id == (node_id << 5 | 0x01):  # 0x01: Heartbeat
            error, state, result, traj_done = struct.unpack('<IBBB', bytes(msg.data[:7]))
            if state == 8:  # 8: AxisState.CLOSED_LOOP_CONTROL
                break

    try:
        while True:
            # 發送速度命令給 ODrive
            bus.send(can.Message(
                arbitration_id=(node_id << 5 | 0x0c),  # 0x0d: Set_Input_Vel
                data=struct.pack('<f', position),  
                is_extended_id=False
            ))

            # 非阻塞接收 CAN 消息
            while True:
                msg = bus.recv(timeout=0.001)  # 設置極短超時時間
                if not msg:
                    break
                if msg and msg.arbitration_id == (node_id << 5 | 0x09):  # 0x09: Get_Encoder_Estimates
                    pos, vel = struct.unpack('<ff', bytes(msg.data))
                    print(f"pos: {pos:.3f} [turns]")

            # 減少發送命令的延遲
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting CAN Bus handler...")

# === ODRIVE ======================================================================================================================================================

if __name__ == "__main__":
    # zed_deepsort(on_update=handle_coords)
    setup_can_interface()
    joystick_thread = threading.Thread(target=joystick_control, daemon=True)
    drivemotor_thread = threading.Thread(target=drivemotor, daemon=True)
    steermotor_thread = threading.Thread(target=steermotor, daemon=True)

    zed_thread = threading.Thread(target=zed_deepsort, args=(handle_coords,), daemon=True)
    zed_thread.start()

    joystick_thread.start()
    drivemotor_thread.start()
    steermotor_thread.start()

    # 等待手柄控制線程完成
    joystick_thread.join()
