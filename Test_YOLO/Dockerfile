# JetPack 6.0-compatible ML container
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Install libraries required by ZED SDK + OpenCV GUI + YOLOv8
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
    wget \
    unzip \
    curl \
    sudo \
    udev \
    pkg-config \
    tzdata \
    python3-dev \
    python3-pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install YOLOv8 + compatible numpy
RUN pip3 install --upgrade pip && \
    pip3 install ultralytics "numpy<2.0"

RUN apt-get update && apt-get install -y libgtk2.0-dev libgtk-3-dev


# For ZED Python API (mounted externally at runtime)
ENV PYTHONPATH=/usr/local/zed/python

# Set working directory
WORKDIR /app

CMD ["bash"]
