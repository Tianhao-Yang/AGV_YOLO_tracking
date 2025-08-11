## This code will show the left, right and the depth view
import pyzed.sl as sl
import cv2
import numpy as np
import datetime

# 建立 ZED 相機物件
zed = sl.Camera()

# 初始化設定
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 15  # Lower frame rate (options: 15, 30, 60 depending on resolution)
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # 可選 NONE, PERFORMANCE, QUALITY, ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER

# 開啟相機
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("無法開啟 ZED 相機")
    exit(1)

# 建立容器
left_image = sl.Mat()
right_image = sl.Mat()
depth_map = sl.Mat()

# 取得畫面尺寸
image_size = zed.get_camera_information().camera_configuration.resolution
width = image_size.width
height = image_size.height

# 設定錄影參數（錄左視圖）
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或 'mp4v' 用於 .mp4
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_out = cv2.VideoWriter(f"zed_left_{timestamp}.avi", fourcc, 30.0, (width, height))

print("按下 'q' 鍵退出")

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # 抓左、右、深度
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        # 轉成 OpenCV 格式
        left = left_image.get_data()
        right = right_image.get_data()
        depth = depth_map.get_data()

        # 將深度圖轉成視覺化灰階
        # 將深度圖中的無效值濾除（NaN、inf、太遠）
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        depth[depth > 5000] = 0  # 單位是 mm，可依需要調整

        # 將深度圖轉為灰階並上色
        # 將深度值 clip 在 0~5000mm，再手動轉為 0~255 灰階
        depth_fixed = np.clip(depth, 0, 5000)
        depth_8bit = ((depth_fixed / 5000.0) * 255.0).astype(np.uint8)
        depth_display = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)



        # 顯示視窗
        cv2.imshow("Left View", left)
        cv2.imshow("Right View", right)
        cv2.imshow("Depth View", depth_display)

        # 寫入影片
        video_out.write(cv2.cvtColor(left, cv2.COLOR_RGBA2BGR))

    # 結束條件：按下 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理資源
video_out.release()
zed.close()
cv2.destroyAllWindows()
print("已關閉並儲存錄影檔案 ✅")
