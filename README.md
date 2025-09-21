Pitch AI MVP

This repository contains an early-stage prototype for a pitching analysis tool. It uses YOLOv8 pose estimation to detect skeleton keypoints in video clips, overlay those points, and calculate basic metrics such as joint angles and stride estimates. A Streamlit app provides the interface for uploading videos and viewing results.

Current State

Working

App launches locally at http://localhost:8501

Model loads when weights are cached

Skeleton overlay works on compatible .mp4 clips

Basic angle calculations (shoulder, elbow, hip)

Not finished

Crashes with “too many open files” if model reloads every frame

.MOV iPhone clips need manual conversion to .mp4

No packaged test videos or sample outputs

UI sometimes fails to render if backend crashes

How to Run

On macOS (M1 tested):
