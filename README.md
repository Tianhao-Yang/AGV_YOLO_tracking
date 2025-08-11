# AGV-YOLO Human Tracking System

This project integrates a ZED stereo camera with an Autonomous Guided Vehicle (AGV) to perform real-time human detection and tracking.

## Overview
- **ZED Camera**: Captures 3D depth and RGB data from the environment.
- **YOLO**: Performs real-time human detection from the ZED video feed.
- **Docker Acceleration**: All detection and tracking processes are containerized and GPU-accelerated using NVIDIA Docker.
- **DeepSORT**: Assigns IDs to detected humans, maintains tracking even with temporary occlusions, and provides consistent target identification.
- **Vehicle Control**: The DeepSORT tracking output is converted into control signals that are sent to the AGV, enabling it to follow the detected human autonomously.

## Workflow
1. Capture frames using the ZED camera.
2. Run YOLO inside a GPU-enabled Docker container for human detection.
3. Pass detection results to DeepSORT for multi-frame tracking.
4. Calculate the target's position and send steering/speed commands to the AGV via its motor control interface.

## Features
- Real-time human detection and tracking.
- Robust target ID persistence with DeepSORT.
- Fully containerized and GPU-accelerated pipeline.
- Autonomous AGV navigation based on target position.

## Requirements
- NVIDIA Jetson CPU
- NVIDIA CUDA GPU
- ZED SDK
- Docker with NVIDIA Container Toolkit
- YOLOv8
- DeepSORT

