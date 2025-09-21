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
# go to project folder
cd ~/Downloads/pitch_ai_mvp

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# download model weights manually
mkdir -p ~/.cache/ultralytics
curl -L -o ~/.cache/ultralytics/yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

# convert .MOV video to .mp4 if needed
ffmpeg -i ~/Downloads/IMG_0.MOV -vcodec libx264 -acodec aac ~/Downloads/IMG_0_fixed.mp4

# run the app
ulimit -n 4096
streamlit run app.py --server.port 8501

Then open http://localhost:8501 in your browser, upload a video, and click Analyze.


Issues and Next Steps

Model reloading: Needs to be initialized once and reused, not on every frame.

Video input: Add built-in conversion so .MOV works automatically.

Error handling: Show errors in the UI instead of silently failing.

Deployment: Local only right now. Needs cloud or containerized demo.

Code cleanup: Add .gitignore to exclude .venv, caches, and model files.




    





















