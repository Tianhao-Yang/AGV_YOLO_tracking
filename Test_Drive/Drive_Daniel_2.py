import pyzed.sl as sl
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import pygame
import serial
import time
import can
import struct
import threading
import subprocess
import sys
import pexpect
# === Track the previous and current target position in each frame ===
previous_x_mm = None
previous_y_mm = None
previous_time = None
wheel_radius = 655.0 / 2.0 # 單位：mm

# === Back wheel parameters ===
wheel_radius_m = 0.3275 # [m] 後輪半徑
max_ang_accel = 0.1 # [rad/s²] 馬達角加速度上限
#previous_angular_velocity = 0.0 # 上一幀馬達角速度命令



# === ZED ======================================================================================================================================================
# === Global variables for click and selected object ===
clicked_position = None
selected_track_id = None

# === Mouse click callback ===
def mouse_callback(event, x, y, flags, param):
 global clicked_position, selected_track_id
 if event == cv2.EVENT_LBUTTONDOWN:
 clicked_position = (x, y)
 selected_track_id = None # Reset on new click
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
 global zed

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
 continue # 略過這一 frame

 # 1. 一波抓影像、深度、點雲資料（來源：ZED SDK 使用指南）🧠
 zed.retrieve_image(image_zed, sl.VIEW.LEFT)
 zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
 zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # 提供 X, Y, Z 座標與 RGBA 色彩 :contentReference[oaicite:5]{index=5}

 frame = image_zed.get_data()
 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

 frame_h, frame_w = frame_rgb.shape[:2]
 center_x = frame_w // 2
 # 2. 畫面中軸紅線（像素對齊 camera self-axis）
 cv2.line(frame_rgb,
 (center_x, 0),
 (center_x, frame_h - 1),
 color=(255, 0, 0), # RGB 紅色
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
 f"3D:({x_mm:.1f},{y_mm:.1f},{z_mm3:.1f})mm "
 f"px_off={offset_px}px({dir_px}) "
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

# === 全域變數 ===
latest_distance = None
latest_offset = None
previous_position = None
current_position = None
previous_time = None
current_time = None

def handle_coords(track_id, x_mm, y_mm, z_mm):
 global latest_distance, latest_offset
 global previous_position, current_position
 global previous_time, current_time
 global dx
 global dz
 global t_frame

 # === 更新目前位置與距離 ===
 latest_distance = z_mm / 1000.0
 latest_offset = x_mm / 1000.0
 current_position = (x_mm, y_mm, z_mm)
 current_time = time.time()

 # === 計算 frame-to-frame 位移 ===
 if previous_position is not None and previous_time is not None:
 dx = (current_position[0] - previous_position[0]) / 1000.0 # [m] # displacement of the target in offset direction in every frame
 dz = (current_position[2] - previous_position[2]) / 1000.0 # [m] # displacement of the target in depth direction in every frame
 t_frame = current_time - previous_time # time interval 
 if t_frame == 0:
 t_frame = 0.05 #default t_frame

 if t_frame > 0:
 vx = dx / t_frame
 vz = dz / t_frame
 else:
 vx = vz = 0.0

 print(f"[Displacement] dx={dx:.3f}m, dz={dz:.3f}m | [Speed] vx={vx:.3f}m/s, vz={vz:.3f}m/s")

 else:
 print(f"[Init] First position: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm")

 # === 更新為下一次的前一幀資料 ===
 previous_position = current_position
 previous_time = current_time


# === PID 控制器類別 ======================================================================================================================================================

# === PID 控制器類別 ===
class PID:
 def __init__(self, kp, ki, kd, output_limit):
 """
 初始化 PID 控制器。
 :param kp: 比例增益
 :param ki: 積分增益
 :param kd: 微分增益
 :param output_limit: 輸出限制（最大控制量）
 """
 self.kp = kp
 self.ki = ki
 self.kd = kd
 self.output_limit = output_limit

 self.integral = 0.0
 self.prev_error = 0.0

 def compute(self, error, t_frame):
 """
 計算 PID 輸出。
 :param error: 當前誤差
 :param t_frame: 時間差（秒）
 :return: 控制輸出，已限制在 output_limit 範圍內
 """
 self.integral += error * t_frame
 derivative = (error - self.prev_error) / t_frame if t_frame > 0 else 0.0

 output = (
 self.kp * error +
 self.ki * self.integral +
 self.kd * derivative
 )

 self.prev_error = error

 # 限制輸出大小
 return max(min(output, self.output_limit), -self.output_limit)
# === 初始化參數 ===
target_distance = 1.3 # 預定距離 (公尺)
distance_threshold = 0.15 # 距離閥值 (公尺)
offset_threshold = 0.2 # 偏移閥值 (公尺)
hold_time = 1.0 # 閥值內/外持續時間 (秒)
const = 0.1
#dt = 0.05 # 固定 50 毫秒 (20 Hz 控制頻率)


# 兩個馬達 PID 各自設定 (可調整參數)
motor_A_pid = PID(kp=0.8, ki=0.02, kd=0.05, output_limit=20) # 馬達 A (距離控制)
motor_B_pid = PID(kp=0.1, ki=0.00, kd=0.03, output_limit=0.2) # 馬達 B (位置控制)

brake_state = True
last_outside_threshold_time = time.time()
last_inside_threshold_time = time.time()

###########################################################################################################################################################
# == 控制邏輯 =============================================================================================================================================
# --- Globals for tracking previous velocity ---
prev_target_velocity = None # [m/s]
prev_timestamp_accel = None # [s]
t_1, t_2, t_3, t_4 = 0.0, 0.0, 0.0,0.0 # [m/s]
previous_angular_velocity = 0.0 # 上一幀馬達角速度命令


alpha_vel = 3 # smoothing factor (0.1 ~ 0.3 common)
velocity_prev = 0.0

def ema(value):
 global velocity_prev
 velocity_prev = alpha_vel * value + (1 - alpha_vel) * velocity_prev
 return velocity_prev





# --- Brake helper ---
# ====== Tunables / globals you likely already have ======
wheel_radius_m = 0.3275
target_distance = 1.3
distance_threshold = 0.2
hold_time = 0.5 # s inside/outside band to toggle brake

max_ang_accel = 3.0 # [rad/s^2] motor angular accel cap
previous_angular_velocity = 0.0

# Brake/timer state
brake_state = False
last_inside_threshold_time = None
last_outside_threshold_time = None
last_still_time = None

# EMA (you already defined this; keep your version if different)
alpha_vel = 0.2
_velocity_prev_internal = 0.0
def ema(value):
 global _velocity_prev_internal
 _velocity_prev_internal = alpha_vel * value + (1 - alpha_vel) * _velocity_prev_internal
 return _velocity_prev_internal


# ====== Helper: update brake with band + stillness ======
def update_brake_state(current_position_z, dz, t_frame, now):
 """
 Uses distance band, hold timers, and "target stillness" to set brake_state.
 Globals used/updated:
 brake_state, target_distance, distance_threshold, hold_time,
 last_inside_threshold_time, last_outside_threshold_time,
 last_still_time
 """
 global brake_state, last_inside_threshold_time, last_outside_threshold_time, last_still_time

 EPS = 1e-6
 V_DEADBAND = 0.03 # [m/s] consider target "still" below this
 STILL_HOLD = 0.5 # [s] continuous stillness before engaging brake in-band

 # init timers
 if last_inside_threshold_time is None:
 last_inside_threshold_time = now
 if last_outside_threshold_time is None:
 last_outside_threshold_time = now
 if last_still_time is None:
 last_still_time = now

 # band check
 dist_err = current_position_z - target_distance
 in_band = abs(dist_err) <= distance_threshold

 # stillness check
 v_inst = abs(dz) / max(t_frame, EPS)
 is_still = v_inst < V_DEADBAND
 if not is_still:
 last_still_time = now # reset when motion detected

 # braking rules
 if in_band:
 # condition A: stayed in band long enough
 if now - last_inside_threshold_time >= hold_time:
 brake_state = True
 # condition B: target is still in band long enough
 if is_still and (now - last_still_time >= STILL_HOLD):
 brake_state = True
 # refresh inside-band timer while we're still counting
 if now - last_inside_threshold_time < hold_time:
 last_inside_threshold_time = now
 else:
 # out of band → prepare to release brake after hold_time
 if now - last_outside_threshold_time >= hold_time:
 brake_state = False
 else:
 last_outside_threshold_time = now

 return brake_state


# ====== Main: drive motor ======
def drive_motor(dz, t_frame, current_position_z):
 """
 dz: target depth displacement this frame [m] (+ away, - toward)
 t_frame: control period [s]
 current_position_z: current distance to target [m]
 returns: (motor_omega_cmd [rad/s], brake_cmd [bool])
 """
 global previous_angular_velocity, brake_state

 EPS = 1e-6
 GEAR = 12.5
 OMEGA_MAX = 30.0 # final *hard* clamp [rad/s] at motor
 GAIN = 1.0 # keep at 1.0 unless you really want extra scaling

 now = time.time()

 # 1) Update brake state (band + stillness)
 brake_state = update_brake_state(current_position_z, dz, t_frame, now)

 # 2) Feedforward target omega from dz (zero if braking)
 if brake_state:
 omega_target = 0.0
 else:
 v_target = dz / max(t_frame, EPS) # [m/s]
 wheel_omega = v_target / max(wheel_radius_m, EPS) # [rad/s] at wheel
 omega_target = wheel_omega * GEAR # [rad/s] at motor

 # 3) Slew-rate limit (acceleration cap)
 max_delta = max_ang_accel * t_frame # [rad/s] allowed change this tick
 delta = np.clip(omega_target - previous_angular_velocity, -max_delta, max_delta)
 omega_slewed = previous_angular_velocity + delta

 # 4) Filter / gain, THEN clamp (prevents 30 → 85 issue)
 omega_filtered = ema(omega_slewed) * GAIN
 omega_cmd = float(np.clip(omega_filtered, -OMEGA_MAX, OMEGA_MAX))

 # 5) Update state; brake cmd true when braking and essentially stopped
 previous_angular_velocity = omega_cmd
 brake_cmd = bool(brake_state and abs(omega_cmd) < 1e-3)

 return omega_cmd, brake_cmd



# === 全域狀態 ===
target_depth_m = None
target_offset_m = None
prev_target_pos_mm = None
curr_target_pos_mm = None
prev_timestamp = None
curr_timestamp = None
frame_dx_m = 0.0
frame_dz_m = 0.0
frame_dt_sec = 0.0

previous_motor_angle_deg = 0.0 # 新增：上一幀的轉向馬達角度命令
MAX_STEER_ANGLE = 25.0 # 馬達角度最大限制（度）

# === 常數參數 ===
CAMERA_TO_FRONT_AXLE = 0.23 # 相機到前軸距離（公尺）

# === 資料對應：平均輪胎角度（度）→ 馬達角度（度）
avg_wheel_angles = np.array([
 -25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,
 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
])
steering_motor_angles = np.array([
 25.343,24.334,23.324,22.313,21.302,20.291,19.279,18.267,17.254,
 16.241,15.228,14.214,13.2,12.186,11.171,10.156,9.1413,8.1261,7.1108,6.0952,
 5.0795,4.0638,3.0479,2.032,1.016,0,-1.016,-2.032,-3.0479,-4.0638,-5.0795,-6.0952,-7.1108,
 -8.1261,-9.1413,-10.156,-11.171,-12.186,-13.2,-14.214,-15.228,-16.241,-17.254,-18.267,-19.279,-20.291,-21.302,-22.313,-23.324,-24.334,-25.343
])
avg_to_motor_angle = interp1d(avg_wheel_angles, steering_motor_angles, kind='linear', fill_value='extrapolate')


def get_steering_angle(x_mm, z_mm):
 """
 Inputs:
 x_mm: lateral offset of target from camera center (mm)
 z_mm: forward distance to target (mm)
 Returns:
 steering motor command angle in degrees (slew-limited, clamped)
 """
 import time
 import numpy as np

 global target_depth_m, target_offset_m
 global prev_timestamp, curr_timestamp, frame_dt_sec
 global previous_motor_angle_deg

 # --- config (local to keep this function self-contained) ---
 MAX_STEER_RATE_DEG_S = 120.0 # deg/s slew limit ("oblique wave")

 # --- timestamps for per-frame dt ---
 curr_timestamp = time.time()
 if prev_timestamp is None:
 frame_dt_sec = 0.05
 else:
 frame_dt_sec = max(curr_timestamp - prev_timestamp, 0.01)
 prev_timestamp = curr_timestamp

 # --- geometry in meters ---
 target_offset_m = (x_mm or 0.0) / 1000.0
 target_depth_m = (z_mm or 0.0) / 1000.0
 forward_m = target_depth_m - CAMERA_TO_FRONT_AXLE

 # invalid geometry → hold last command
 if not np.isfinite(forward_m) or forward_m <= 0.0:
 return previous_motor_angle_deg

 # 1) theoretical avg wheel angle [deg]
 avg_wheel_angle_deg = float(np.degrees(np.arctan2(target_offset_m, forward_m)))

 # 2) interpolate avg wheel angle -> motor angle [deg]
 target_motor_deg = float(avg_to_motor_angle(avg_wheel_angle_deg))

 # 3) clamp to mechanical limits
 target_motor_deg = float(np.clip(target_motor_deg, -MAX_STEER_ANGLE, MAX_STEER_ANGLE))

 # 4) slew-limit ("oblique wave") per frame
 max_delta = MAX_STEER_RATE_DEG_S * frame_dt_sec
 delta = np.clip(target_motor_deg - previous_motor_angle_deg, -max_delta, max_delta) 
 cmd_deg = (previous_motor_angle_deg + float(delta)) / 360 * 1.5 #編碼器速比換算

 # update state & return
 previous_motor_angle_deg = cmd_deg
 return cmd_deg

# ==== EXAMPLE USAGE ================================================================
# controller = SteeringController()
# while True:
# # from ZED callback (selected target):
# motor_deg, motor_turns = controller.compute(x_mm, z_mm)
# position = motor_turns # <-- send to your steermotor thread (Set_Input_Pos)
# print(f"[STEER] geom-> {motor_deg:.2f}° ({motor_turns:.3f} turns)")



# === PID 控制器類別 ======================================================================================================================================================

# === ODRIVE ======================================================================================================================================================

# 初始化 pygame 手柄模組
pygame.init()
pygame.joystick.init()

# 設定 UART 連接
ser = serial.Serial('/dev/ttyACM0', 9600) # 根據實際情況調整端口名稱
time.sleep(2) # 等待 UART 連接穩定

# 連接手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Connected to {joystick.get_name()}")

# 馬達控制參數
velocity = 0.0 # 初始速度
max_velocity = 20.0
min_velocity = -20.0
# velocity_increment = 1.0

position = 0.0 # 初始位置
max_position = 0.2 #極限50 degree
min_position = -0.2 #極限-50 degree 要記得1.5倍皮帶輪加速轉換 編碼器讀值為-75 degree
# position_increment = 0.02

led_on = False # LED 狀態

def setup_can_interface():
 try:
 print("Setting up CAN interface...")
 subprocess.run(['ip', 'link', 'set', 'can0', 'down'], check=False) # 忽略失敗
 subprocess.run(['ip', 'link', 'set', 'can0', 'up', 'type', 'can', 'bitrate', '250000'], check=True)
 print("CAN interface setup complete.")
 except subprocess.CalledProcessError as e:
 print(f"Failed to set up CAN interface: {e}")
 sys.exit(1)

# 控制 LED 的函數
def control_led(state):
 global led_on
 if state == 'on' and not led_on:
 ser.write(b'1') # 發送開啟 LED 指令
 led_on = True
 print("LED 開啟 (煞車)")
 elif state == 'off' and led_on:
 ser.write(b'0') # 發送關閉 LED 指令
 led_on = False
 print("LED 關閉 (解除煞車)")
 time.sleep(0.05)

# PS2 控制線程
def joystick_control():
 global velocity, position
 velocity = 0
 position = 0
 mode_active = False # 模式是否啟動

 # ====== super loop ================================================================================================================================
 def mode_loop():
 """進入特殊模式後的持續執行邏輯"""
 print("進入模式中...")
 while True:
 pygame.event.pump()
 buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

 # === 模式退出條件 ===
 if buttons[1] == 1 or buttons[3] == 1: # 按下 B鍵 或 X鍵(煞車)
 print("退出模式，回到手動控制")
 break
 # measured_distance, measured_offset = handle_coords(track_id, x_mm, y_mm, z_mm)
 x_mm = latest_offset * 1000
 z_mm = latest_distance * 1000
 y_mm = 0.0 # or use tracked value if needed
 if latest_distance is not None and latest_offset is not None:
 speed_cmd, brake_cmd = drive_motor(dz,t_frame,z_mm)
 steer_cmd = get_steering_angle(x_mm, z_mm)
 # measured_distance, measured_offset = handle_coords(track_id, x_mm, y_mm, z_mm)
 # if latest_distance is not None and latest_offset is not None:
 # # speed_cmd, brake_cmd = drive_motor(dz, t_frame)
 # sensors_data = sl.SensorsData()
 # zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)

 # # 假設你已經取得 dz, t_frame
 # speed_cmd, brake_cmd = drive_motor(dz, sensors_data,t_frame)

 # # Convert back to mm
 # x_mm = latest_offset * 1000
 # z_mm = latest_distance * 1000
 # y_mm = 0.0 # or use tracked value if needed

 # # Get ZED IMU data
 # sensors_data = sl.SensorsData()
 # zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)

 # # Compute motor angle command
 # steer_cmd = get_steering_motor_angle(x_mm, y_mm, z_mm, sensors_data)



 # speed_cmd, steer_cmd, brake_cmd = control_loop(latest_distance, latest_offset, t_frame)
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
 mode_loop() # 進入模式
 mode_active = False # 模式結束後回到手動

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
 node_id = 0 # 必須與 <odrv>.axis0.config.can.node_id 匹配，默認為 0
 global linear_velocity
 bus = can.interface.Bus("can0", interface="socketcan")

 # 清空 CAN RX 緩衝區，以確保沒有舊的消息
 while not (bus.recv(timeout=0) is None):
 pass

 # 將軸設置為閉環控制狀態
 bus.send(can.Message(
 arbitration_id=(node_id << 5 | 0x07), # 0x07: Set_Axis_State
 data=struct.pack('<I', 8), # 8: AxisState.CLOSED_LOOP_CONTROL
 is_extended_id=False
 ))

 # 通過心跳消息等待軸進入閉環控制狀態
 for msg in bus:
 if msg.arbitration_id == (node_id << 5 | 0x01): # 0x01: Heartbeat
 error, state, result, traj_done = struct.unpack('<IBBB', bytes(msg.data[:7]))
 if state == 8: # 8: AxisState.CLOSED_LOOP_CONTROL
 break

 try:
 while True:
 # 發送速度命令給 ODrive
 bus.send(can.Message(
 arbitration_id=(node_id << 5 | 0x0d), # 0x0d: Set_Input_Vel
 data=struct.pack('<ff', velocity, 0.0), # 速度和扭矩前饋
 is_extended_id=False
 ))

 # 非阻塞接收 CAN 消息
 while True:
 msg = bus.recv(timeout=0.001) # 設置極短超時時間
 if not msg:
 break
 if msg.arbitration_id == (node_id << 5 | 0x09): # 0x09: Get_Encoder_Estimates
 pos, vel = struct.unpack('<ff', bytes(msg.data))
 print(f"vel: {vel:.3f} [turns/s]")
 linear_velocity = vel

 # 減少發送命令的延遲
 time.sleep(0.01)
 except KeyboardInterrupt:
 print("Exiting CAN Bus handler...")

# 轉向馬達線程
def steermotor():
 node_id = 1 # 必須與 <odrv>.axis0.config.can.node_id 匹配，默認為 0

 bus = can.interface.Bus("can0", interface="socketcan")

 # 清空 CAN RX 緩衝區，以確保沒有舊的消息
 while not (bus.recv(timeout=0) is None):
 pass

 # 將軸設置為閉環控制狀態
 bus.send(can.Message(
 arbitration_id=(node_id << 5 | 0x07), # 0x07: Set_Axis_State
 data=struct.pack('<I', 8), # 8: AxisState.CLOSED_LOOP_CONTROL
 is_extended_id=False
 ))

 # 通過心跳消息等待軸進入閉環控制狀態
 for msg in bus:
 if msg.arbitration_id == (node_id << 5 | 0x01): # 0x01: Heartbeat
 error, state, result, traj_done = struct.unpack('<IBBB', bytes(msg.data[:7]))
 if state == 8: # 8: AxisState.CLOSED_LOOP_CONTROL
 break

 try:
 while True:
 # 發送速度命令給 ODrive
 bus.send(can.Message(
 arbitration_id=(node_id << 5 | 0x0c), # 0x0c Set_Input_Pos
 data=struct.pack('<f', position), 
 is_extended_id=False
 ))

 # 非阻塞接收 CAN 消息
 while True:
 msg = bus.recv(timeout=0.001) # 設置極短超時時間
 if not msg:
 break
 if msg and msg.arbitration_id == (node_id << 5 | 0x09): # 0x09: Get_Encoder_Estimates
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
