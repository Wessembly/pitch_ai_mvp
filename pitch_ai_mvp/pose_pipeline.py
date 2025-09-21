from ultralytics import YOLO
import cv2, numpy as np
from smoothing import OneEuroFilter
import ffmpeg

KEYPOINT_NAMES = [
    # YOLOv8-pose uses 17 by default; some weights have 33. We map what we get.
    # We'll handle variable length robustly.
]

# Load YOLO model once
_model = YOLO("yolov8n-pose.pt")

def _euro():
    return OneEuroFilter(min_cutoff=1.7, beta=0.3, d_cutoff=1.0)

def analyze_video(path, pitcher_height_cm=185, fps_override=None):
    cap = cv2.VideoCapture(path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filters = {}
    series = {"angles": [], "points": [], "scale_px_per_m": None, "fps": fps}
    scale_px_per_m = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = path.replace(".mp4", "_annot.mp4").replace(".mov","_annot.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_idx = 0

    # Run inference once, stream results
    for res in _model(path, stream=True, verbose=False):
        frame = res.orig_img
        if len(res.keypoints) == 0:
            writer.write(frame); frame_idx += 1; continue

        kps = res.keypoints.xy[0].cpu().numpy()
        if res.keypoints.conf is not None:
            conf = res.keypoints.conf[0].cpu().numpy()
        else:
            conf = np.ones(len(kps))

        # init filters
        for i in range(len(kps)):
            if i not in filters: filters[i] = (_euro(), _euro())

        # smooth keypoints
        smoothed = []
        t = frame_idx / fps
        for i,(x,y) in enumerate(kps):
            fx, fy = filters[i]
            smoothed.append([fx(x,t), fy(y,t)])
        smoothed = np.array(smoothed)

        # estimate scale once
        if scale_px_per_m is None and len(smoothed)>0:
            top = smoothed[:,1].min(); bot = smoothed[:,1].max()
            px_height = max(1.0, bot-top)
            scale_px_per_m = px_height / (0.96*(pitcher_height_cm/100))
            series["scale_px_per_m"] = scale_px_per_m

        # draw skeleton
        for p in smoothed:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
        for i in range(len(smoothed)-1):
            p1 = tuple(map(int, smoothed[i]))
            p2 = tuple(map(int, smoothed[i+1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 1)

        # simple angle calc
        def angle(a,b,c):
            ba = a-b; bc = c-b
            cosang = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
            return float(np.degrees(np.arccos(np.clip(cosang,-1,1))))
        def get(i): 
            return smoothed[i] if i < len(smoothed) else None

        left = {"elbow": None, "shoulder": None, "hip": None}
        if get(7) is not None and get(5) is not None and get(9) is not None:
            left["elbow"] = angle(get(5), get(7), get(9))
        if get(6) is not None and get(5) is not None and get(11) is not None:
            left["shoulder"] = angle(get(6), get(5), get(11))
        if get(11) is not None and get(12) is not None and get(13) is not None:
            left["hip"] = angle(get(12), get(11), get(13))

        series["angles"].append(left)
        series["points"].append(smoothed.tolist())

        writer.write(frame)
        frame_idx += 1

    cap.release(); writer.release()
    return out_path, series
