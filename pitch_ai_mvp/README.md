
# Pitch AI MVP — Computer Vision Pitching Analyzer

Upload a pitching video → get pose overlay, key event timestamps, and basic mechanics metrics.

## Quickstart
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download a YOLOv8 pose model (lightweight)
# This happens automatically on first run, but you can prefetch:
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# Run Streamlit app
streamlit run app.py
```
## Notes
- Uses YOLOv8-pose for 33 keypoints, basic IOU tracking, and a 1‑Euro filter for smoothing.
- Metrics are naive but useful: stride length, release point height/offset, hip‑shoulder separation proxy, arm slot.
- Event detection via velocity/angle peaks: foot strike, max ER proxy, ball release heuristic.
- This is an MVP. Validate on your own clips and refine thresholds.
