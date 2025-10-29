import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import imutils
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(page_title="Shoplifting Detection", layout="wide")
st.title("🛍️ Shoplifting Detection using YOLOv8")
try:
    import cv2
except ImportError as e:
    raise RuntimeError(
        "⚠️ OpenCV failed to import. If running on Streamlit Cloud, "
        "add 'runtime.txt' with 'python-3.10' or reinstall opencv-python-headless."
    ) from e

# Sidebar configuration
st.sidebar.header("Settings")
weights_path = st.sidebar.text_input("Model Weights Path (.pt)", "best.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
device = "cuda" if torch.cuda.is_available() else "cpu"

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        input_video_path = temp_video.name

    st.video(input_video_path)

    if st.button("Run Detection 🚀"):
        st.write("**Processing video... Please wait.**")

        model = YOLO(weights_path)
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = imutils.resize(frame, width=800)
            results = model.predict(frame, device=device, conf=confidence_threshold, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                color = (0, 255, 255) if cls == 1 else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"Frame {frame_count}/{total_frames}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            out.write(frame)
            progress.progress(frame_count / total_frames)

        cap.release()
        out.release()
        st.success("✅ Processing Complete!")
        st.video(out_path)
else:
    st.info("Please upload a video to begin.")

