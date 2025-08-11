# 基於 JetPack 6.0 ML 容器 (支援 GPU 與 PyTorch)
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 安裝 ZED SDK、OpenCV GUI、YOLOv8、DeepSORT、CAN Bus、Joystick、Serial 所需依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libturbojpeg \
    libegl1 \
    libgles2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libsm6 \
    libxrender1 \
    libusb-1.0-0 \
    libgtk2.0-dev \
    libgtk-3-dev \
    can-utils \
    joystick \
    udev \
    pkg-config \
    wget \
    unzip \
    curl \
    sudo \
    tzdata \
    python3-dev \
    python3-pip \
    python3-serial \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 升級 pip 並安裝 Python 套件
RUN pip3 install --upgrade pip && \
    pip3 install \
        "numpy<2.0" \
        scipy \
        ultralytics \
        pygame \
        python-can \
        pexpect \
        pyserial \
        deep-sort-realtime

# 可選：再次確保 CAN 工具齊全
RUN apt-get update && apt-get install -y iproute2 can-utils

# 設定 ZED Python API (於 runtime 掛載)
ENV PYTHONPATH=/usr/local/zed/python

# 設定工作目錄
WORKDIR /app

# 預設進入點（可在 docker run 覆蓋）
CMD ["bash"]
