# JetPack 6.0-compatible ML container
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV OPENCV_VERSION=4.8.0

# ---------------------------
# 1. Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev libopenexr-dev \
    libatlas-base-dev gfortran \
    libturbojpeg libegl1 libgles2 libgl1-mesa-glx \
    libglib2.0-0 libx11-6 libxext6 libxrandr2 libxinerama1 \
    libxcursor1 libsm6 libxrender1 libusb-1.0-0 \
    wget unzip curl sudo udev tzdata python3-dev python3-pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 2. Upgrade pip + install Python packages
# ---------------------------
RUN pip3 install --upgrade pip && \
# Use NumPy 1.24.4 to avoid np.float errors
    pip3 install numpy==1.24.4 ultralytics deep_sort_realtime


# ---------------------------
# 3. Fix NumPy 2.x deprecations (np.float)
# ---------------------------
RUN PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    echo 'import numpy as np\nnp.float = np.float64\nnp.float_ = np.float64' > $PYTHON_SITE/sitecustomize.py

# ---------------------------
# 4. Build OpenCV with minimal options
# ---------------------------
WORKDIR /tmp/opencv_build

RUN git clone --depth=1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth=1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
    mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
          -D BUILD_opencv_python3=ON \
          -D PYTHON_EXECUTABLE=$(which python3) \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_opencv_js=OFF \
          -D WITH_QT=OFF \
          -D WITH_GTK=ON \
          -D WITH_OPENGL=OFF \
          ../opencv && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/opencv_build

# ---------------------------
# 5. Set up ZED SDK path
# ---------------------------
ENV PYTHONPATH=/usr/local/zed/python

# ---------------------------
# 6. Set working directory
# ---------------------------
WORKDIR /app
CMD ["bash"]
