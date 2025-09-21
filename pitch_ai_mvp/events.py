
import numpy as np

def detect_events(series):
    # Heuristic event detection based on vertical ankle velocity and elbow/shoulder angles
    fps = series.get("fps", 30)
    pts = series["points"]
    if not pts: return {}

    pts_np = np.array([np.array(p) for p in pts], dtype=object)  # frames x K x 2
    # choose left ankle index 15 if present (COCO)
    idx_ankle = 15 if pts_np[0].shape[0] > 15 else None
    y = []
    for f in range(len(pts_np)):
        if idx_ankle is not None and idx_ankle < len(pts_np[f]):
            y.append(pts_np[f][idx_ankle][1])
        else:
            y.append(np.nan)
    y = np.array(y, dtype=float)
    dy = np.gradient(y)
    # foot strike ~ local minimum in dy after stride, ball release ~ elbow angle velocity peak
    angles = [a.get("elbow") or np.nan for a in series["angles"]]
    ang = np.array(angles, dtype=float)
    dang = np.gradient(ang)

    def peak(arr):
        if np.all(np.isnan(arr)): return None
        idx = np.nanargmin(arr) if arr.var() > 0 else None
        return int(idx) if idx is not None else None

    foot_strike = peak(dy)
    release = int(np.nanargmax(dang)) if not np.isnan(dang).all() else None

    return {"fps": fps, "foot_strike_frame": foot_strike, "ball_release_frame": release}
