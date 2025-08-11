# Build args (example: 36.2.0 and l4t-ml or l4t-base)
ARG L4T_MAJOR_VERSION
ARG L4T_MINOR_VERSION
ARG L4T_PATCH_VERSION
ARG L4T_BASE_IMAGE
ARG ZED_SDK_MAJOR
ARG ZED_SDK_MINOR

FROM nvcr.io/nvidia/${L4T_BASE_IMAGE}:r${L4T_MAJOR_VERSION}.${L4T_MINOR_VERSION}.${L4T_PATCH_VERSION}

ARG L4T_MAJOR_VERSION
ARG L4T_MINOR_VERSION
ARG L4T_PATCH_VERSION
ARG ZED_SDK_MAJOR
ARG ZED_SDK_MINOR

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV LOGNAME=root

# --- Base deps + ZED SDK ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates lsb-release less zstd udev sudo apt-transport-https \
    # GUI + OpenCV runtime libs
    libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libxrandr2 libgtk-3-0 \
    # GStreamer for VideoWriter / camera pipelines
    libgstreamer1.0-0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    && echo "# R${L4T_MAJOR_VERSION} (release), REVISION: ${L4T_MINOR_VERSION}.${L4T_PATCH_VERSION}" > /etc/nv_tegra_release \
    && wget -q --no-check-certificate -O ZED_SDK_Linux.run \
         https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/l4t${L4T_MAJOR_VERSION}.${L4T_MINOR_VERSION}/jetsons \
    && chmod +x ZED_SDK_Linux.run \
    && ./ZED_SDK_Linux.run silent runtime_only skip_drivers \
    && rm -rf /usr/local/zed/resources/* \
    && rm -rf ZED_SDK_Linux.run \
    && rm -rf /var/lib/apt/lists/*

# --- Python + ZED Python API ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools python3-dev build-essential \
    && python3 -m pip install --no-cache-dir -U pip \
    # grab Stereolabs official helper and install pyzed
    && wget -q https://download.stereolabs.com/zedsdk/pyzed -O /usr/local/zed/get_python_api.py \
    && python3 -m pip install --no-cache-dir requests \
    && python3 /usr/local/zed/get_python_api.py --force \
    # optional: wheels/cython on aarch64 are handled by the helper; keep minimal
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# For ZED streaming features inside container
RUN ln -sf /usr/lib/aarch64-linux-gnu/tegra/libv4l2.so.0 /usr/lib/aarch64-linux-gnu/libv4l2.so

WORKDIR /usr/local/zed
