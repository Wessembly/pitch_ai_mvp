
import numpy as np
import pandas as pd

def compute_metrics_table(series, events, pitcher_height_cm):
    pts = series["points"]
    scale = series.get("scale_px_per_m") or 500.0  # px per meter
    if not pts: 
        return pd.DataFrame([{"metric":"none","value":0,"unit":""}])
    # stride length = distance heel-to-heel at foot strike (approx ankles 15 and 16 if present)
    f = events.get("foot_strike_frame") or int(len(pts)/2)
    P = np.array(pts[min(f, len(pts)-1)])
    def get(i): 
        return P[i] if i < len(P) else None

    aL, aR = get(15), get(16)
    stride_px = np.linalg.norm(aL - aR) if aL is not None and aR is not None else np.nan
    stride_m = float(stride_px) / scale if stride_px==stride_px else np.nan

    # release point height = wrist y at release (index 9 or 10)
    r = events.get("ball_release_frame") or min(len(pts)-1, f+5)
    R = np.array(pts[r])
    w = R[9] if 9 < len(R) else None
    release_h_m = np.nan
    if w is not None:
        # convert y px to height above ground using frame min/max as rough ground reference
        # naive: height proportional to (maxY - y)
        img_h = max(R[:,1]) - min(R[:,1])
        release_h_m = float((max(R[:,1]) - w[1]) / scale)

    # arm slot proxy = angle of forearm relative to vertical at release (elbow 7, wrist 9)
    arm_slot_deg = np.nan
    if 7 < len(R) and 9 < len(R):
        v = R[9] - R[7]
        arm_slot_deg = float(np.degrees(np.arctan2(v[0], -v[1])))

    data = [
        {"metric":"stride_length","value":round(stride_m,3),"unit":"m"},
        {"metric":"release_height","value":round(release_h_m,3),"unit":"m"},
        {"metric":"arm_slot","value":round(arm_slot_deg,1),"unit":"deg"},
    ]
    return pd.DataFrame(data)
