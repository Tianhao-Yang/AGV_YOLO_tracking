##Drive program
import pygame
import serial
import time
import can
import struct
import threading
import subprocess
import sys
import pexpect

# def setup_can_interface():# set up the communitation
#     """
#     Set up the CAN interface using a system command. 
#     """
#     command = "sudo -S ip link set can0 up type can bitrate 250000" # set up the bitrate 250000
#     try:
#         print("Setting up CAN interface...")
#         child = pexpect.spawn(command, encoding='utf-8')
#         child.expect("password for")
#         child.sendline("chanky.123#")  # 替換為你的密碼
#         child.expect(pexpect.EOF)
#         print("CAN interface setup complete.")
#     except pexpect.exceptions.ExceptionPexpect as e:
#         print(f"Failed to set up CAN interface: {e}")
#         sys.exit(1)


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
max_velocity = 40.0
min_velocity = -40.0
# velocity_increment = 1.0

position = 0.0  # 初始位置
max_position = 0.2 #極限50 degree
min_position = -0.2 #極限-50 degree 要記得1.5倍皮帶輪加速轉換 編碼器讀值為-75 degree
# position_increment = 0.02

led_on = False  # LED 狀態

# setup_can_interface()
def setup_can_interface():
    import subprocess

def setup_can_interface():
    try:
        print("Setting up CAN interface...")
        subprocess.run(
            ['sudo', '/usr/sbin/ip', 'link', 'set', 'can0', 'up', 'type', 'can', 'bitrate', '250000'],
            check=True
        )
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

    try:
        while True:
            pygame.event.pump()  # 更新事件

            # 讀取所有按鈕和方向鍵
            hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]          #????
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())] #????

            # 按鈕 3 (X鍵）(煞車) 控制 LED 與速度
            if buttons[3] == 1:  # 按下
                velocity = 0
                control_led('on')  # 煞車時開啟 LED
                print("煞車")
            elif buttons[3] == 0:  # 放開
                control_led('off')  # 解除煞車時關閉 LED
                print("煞車解除")

            # 用 axis 取代 D-pad 方向鍵控制
            if joystick.get_numaxes() >= 4:
                axis_steer = -joystick.get_axis(0)  # 左右搖桿 X 軸
                axis_drive = joystick.get_axis(3)  # 右搖桿 Y 軸（某些控制器為 trigger）

                # 轉換百分比到對應區間
                position = max(min_position, min(max_position, axis_steer * max_position))
                velocity = max(min_velocity, min(max_velocity, -axis_drive * max_velocity))  # Y 軸通常反向

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
# 啟動程式
if __name__ == "__main__":
    # 初始化 CAN 接口

    setup_can_interface()
    joystick_thread = threading.Thread(target=joystick_control, daemon=True)
    drivemotor_thread = threading.Thread(target=drivemotor, daemon=True)
    steermotor_thread = threading.Thread(target=steermotor, daemon=True)

    joystick_thread.start()
    drivemotor_thread.start()
    steermotor_thread.start()

    # 等待手柄控制線程完成
    joystick_thread.join()
