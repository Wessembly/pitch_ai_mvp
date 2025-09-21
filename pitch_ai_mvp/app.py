
import streamlit as st
import tempfile, os
from pose_pipeline import analyze_video
from metrics import compute_metrics_table
from events import detect_events

st.set_page_config(page_title="Pitch AI MVP", layout="wide")
st.title("Pitch AI — MVP")
st.write("Upload a pitching clip (side view recommended).")

video = st.file_uploader("MP4/MOV", type=["mp4","mov","mkv"])
calib_height = st.number_input("Pitcher height (cm) for scale estimate", min_value=120, max_value=220, value=185)
fps_override = st.number_input("FPS override (0=auto)", min_value=0, max_value=240, value=0)

if st.button("Analyze") and video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.read())
        tmp.flush()
        out_path, series = analyze_video(tmp.name, pitcher_height_cm=calib_height, fps_override=(fps_override or None))
    st.video(out_path)
    events = detect_events(series)
    st.subheader("Detected Events")
    st.json(events)
    st.subheader("Metrics")
    table = compute_metrics_table(series, events, calib_height)
    st.dataframe(table)
    st.download_button("Download annotated video", data=open(out_path,'rb'), file_name=os.path.basename(out_path))
