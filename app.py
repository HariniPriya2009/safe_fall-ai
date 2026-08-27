import os
import tempfile
import urllib.request
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(
    page_title="SafeFall AI - Elderly Fall Detection",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #1B2A2E;
    --muted: #6B7C7A;
    --paper: #F5F6F3;
    --surface: #FFFFFF;
    --primary: #0E6E62;
    --primary-dim: #E3EFEC;
    --alert: #C6433C;
    --alert-dim: #FBEAE8;
    --safe: #2F9E6E;
    --safe-dim: #E8F5EE;
    --line: #E4E6E1;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
[data-testid="stAppViewContainer"] { background: var(--paper); }

/* ---- Header ---- */
.sf-header { padding: 0.25rem 0 1rem 0; }
.sf-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--primary); margin-bottom: 0.35rem;
}
.sf-title {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.6rem;
    line-height: 1.08; color: var(--ink); margin: 0;
}
.sf-subtitle { font-size: 1.02rem; color: var(--muted); margin-top: 0.6rem; max-width: 640px; }

/* Pulse-line signature divider (ECG-style) */
.sf-pulse { width: 100%; height: 34px; margin: 1.1rem 0 0.4rem 0; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
.sf-badge {
    width: 42px; height: 42px; border-radius: 10px; background: var(--primary);
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
    margin-bottom: 0.6rem;
}
.sf-sidebar-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.25rem; margin: 0; }
.sf-sidebar-sub { font-size: 0.82rem; color: var(--muted); margin-top: 0.1rem; }

/* ---- Cards / sections ---- */
.sf-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
    padding: 1.2rem 1.3rem; box-shadow: 0 1px 2px rgba(27,42,46,0.04);
}
.sf-section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}

/* ---- Metric cards ---- */
.sf-metric-row { display: flex; gap: 0.7rem; }
.sf-metric {
    flex: 1; background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
    padding: 0.9rem 1rem;
}
.sf-metric-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
}
.sf-metric-value { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.9rem; color: var(--ink); margin-top: 0.15rem; }
.sf-metric-fall .sf-metric-value { color: var(--alert); }
.sf-metric-normal .sf-metric-value { color: var(--safe); }

/* ---- Alert cards (fall detection) ---- */
.sf-alert {
    border-radius: 12px; padding: 1rem 1.2rem; display: flex; gap: 0.8rem;
    align-items: flex-start; margin: 0.6rem 0;
}
.sf-alert-danger { background: var(--alert-dim); border: 1px solid #EFC3C0; }
.sf-alert-safe { background: var(--safe-dim); border: 1px solid #BFE3D2; }
.sf-alert-icon { font-size: 1.3rem; line-height: 1.4; }
.sf-alert-title { font-weight: 600; font-size: 0.98rem; }
.sf-alert-danger .sf-alert-title { color: var(--alert); }
.sf-alert-safe .sf-alert-title { color: var(--safe); }
.sf-alert-body { font-size: 0.88rem; color: var(--muted); margin-top: 0.15rem; }

/* ---- Prediction result banner ---- */
.sf-result {
    display: flex; align-items: baseline; gap: 0.6rem; padding: 0.85rem 1.1rem;
    background: var(--primary-dim); border-radius: 10px; margin: 0.7rem 0;
}
.sf-result-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--primary); }
.sf-result-value { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.3rem; color: var(--ink); }
.sf-result-conf { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; color: var(--muted); margin-left: auto; }

/* ---- Widgets ---- */
.stButton>button { border-radius: 8px; font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; }
[data-testid="stFileUploaderDropzone"] { border-radius: 10px; }
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "fall_detection_model.h5"
SCALER_PATH = "feature_scaler.pkl"
LABELS_PATH = "label_encoder_classes.npy"
POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
FALL_LABEL = "fall"
ALERT_CONFIDENCE_THRESHOLD = 0.60
VIDEO_FRAME_SAMPLE_RATE = 5  # process every Nth frame for speed

# Hardcoded BlazePose 33-point skeleton connections. NOTE: we define this
# manually instead of importing mp.solutions.pose.POSE_CONNECTIONS because
# the legacy mp.solutions API has been breaking with
# "AttributeError: module 'mediapipe' has no attribute 'solutions'" on
# recent MediaPipe releases. This list is identical to the official one,
# just declared directly so we don't depend on the broken module.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


# ----------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    class_names = np.load(LABELS_PATH, allow_pickle=True)
    return model, scaler, class_names


@st.cache_resource
def get_pose_detector():
    # Download the pose model file on first run if it isn't already in
    # the repo (Streamlit Cloud gets a fresh filesystem each deploy).
    if not os.path.isfile(POSE_MODEL_PATH):
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ----------------------------------------------------------------------
# SESSION STATE (dashboard counters persist across uploads)
# ----------------------------------------------------------------------
if "total_detected" not in st.session_state:
    st.session_state.total_detected = 0
if "fall_count" not in st.session_state:
    st.session_state.fall_count = 0
if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []  # list of dicts: {activity, confidence}


# ----------------------------------------------------------------------
# CORE PREDICTION FUNCTION
# ----------------------------------------------------------------------
def draw_pose_landmarks(image_bgr, landmarks):
    """
    Manually draws the pose skeleton with OpenCV instead of
    mp.solutions.drawing_utils (avoids the broken legacy API).
    """
    h, w = image_bgr.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(image_bgr, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)

    return image_bgr


def predict_frame(frame_bgr, model, scaler, class_names, pose_detector):
    """
    Runs pose detection + classification on a single BGR frame.
    Returns (annotated_frame, predicted_label, confidence) or
    (annotated_frame, None, None) if no pose was detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = pose_detector.detect(mp_image)

    annotated = frame_bgr.copy()

    if not result.pose_landmarks:
        return annotated, None, None

    landmarks = result.pose_landmarks[0]  # first detected person
    annotated = draw_pose_landmarks(annotated, landmarks)

    row = []
    for lm in landmarks:
        visibility = getattr(lm, "visibility", 0.0)
        row.extend([lm.x, lm.y, lm.z, visibility])

    features = np.array(row, dtype="float32").reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)

    probs = model.predict(features_reshaped, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label = str(class_names[pred_idx])

    return annotated, label, confidence


def log_prediction(label, confidence):
    st.session_state.total_detected += 1
    if label.lower() == FALL_LABEL:
        st.session_state.fall_count += 1
    else:
        st.session_state.normal_count += 1
    st.session_state.activity_log.append({"activity": label, "confidence": round(confidence, 3)})


def show_alert_if_fall(label, confidence):
    if label.lower() == FALL_LABEL and confidence >= ALERT_CONFIDENCE_THRESHOLD:
        st.markdown(
            f"""
            <div class="sf-alert sf-alert-danger">
              <div class="sf-alert-icon">🚨</div>
              <div>
                <div class="sf-alert-title">Emergency alert — fall detected</div>
                <div class="sf-alert-body">Confidence {confidence:.1%}. Caregiver notification triggered.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif label.lower() == FALL_LABEL:
        st.markdown(
            f"""
            <div class="sf-alert sf-alert-danger">
              <div class="sf-alert-icon">⚠️</div>
              <div>
                <div class="sf-alert-title">Possible fall — low confidence</div>
                <div class="sf-alert-body">Confidence {confidence:.1%}. Please verify manually.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sf-badge">🚨</div>', unsafe_allow_html=True)
    st.markdown('<p class="sf-sidebar-title">SafeFall AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sf-sidebar-sub">CareVision HealthTech Pvt. Ltd.</p>', unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio("Input source", ["Upload Image", "Upload Video"], label_visibility="visible")
    st.markdown("---")
    if st.button("Reset dashboard counters", width="stretch"):
        st.session_state.total_detected = 0
        st.session_state.fall_count = 0
        st.session_state.normal_count = 0
        st.session_state.activity_log = []
        st.rerun()

# ----------------------------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------------------------
PULSE_SVG = """
<svg class="sf-pulse" viewBox="0 0 600 34" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <polyline points="0,17 140,17 160,17 172,4 184,30 196,10 208,17 240,17 600,17"
    fill="none" stroke="#0E6E62" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

st.markdown(
    f"""
    <div class="sf-header">
      <div class="sf-eyebrow">Computer Vision &middot; Healthcare Monitoring</div>
      <h1 class="sf-title">SafeFall AI</h1>
      <p class="sf-subtitle">Pose-estimation powered activity monitoring for elderly safety.
      Upload a snapshot or video clip to classify activity and detect fall incidents in real time.</p>
      {PULSE_SVG}
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler, class_names = load_artifacts()
    pose_detector = get_pose_detector()
except Exception as e:
    st.error(
        "Could not load model artifacts. Make sure fall_detection_model.h5, "
        "feature_scaler.pkl, and label_encoder_classes.npy are in the repo "
        f"root alongside app.py.\n\nError: {e}"
    )
    st.stop()

col_main, col_dashboard = st.columns([2, 1])

def render_result_banner(label, confidence):
    st.markdown(
        f"""
        <div class="sf-result">
          <span class="sf-result-label">Predicted activity</span>
          <span class="sf-result-value">{label.upper()}</span>
          <span class="sf-result-conf">{confidence:.1%} confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# LEFT COLUMN: upload + prediction
# ----------------------------------------------------------------------
with col_main:
    st.markdown('<div class="sf-section-label">Input</div>', unsafe_allow_html=True)

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated, label, confidence = predict_frame(frame, model, scaler, class_names, pose_detector)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                      caption="Pose detection output", width="stretch")

            if label is None:
                st.markdown(
                    """
                    <div class="sf-alert sf-alert-danger">
                      <div class="sf-alert-icon">🔍</div>
                      <div>
                        <div class="sf-alert-title">No pose detected</div>
                        <div class="sf-alert-body">Try a clearer frame with the full body visible.</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                render_result_banner(label, confidence)
                show_alert_if_fall(label, confidence)
                log_prediction(label, confidence)

    else:  # Upload Video
        uploaded_video = st.file_uploader("Upload a video, supports only .mp4", type=["mp4"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % VIDEO_FRAME_SAMPLE_RATE == 0:
                    annotated, label, confidence = predict_frame(
                        frame, model, scaler, class_names, pose_detector
                    )
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}", width="stretch"
                    )
                    if label is not None:
                        with status_placeholder.container():
                            render_result_banner(label, confidence)
                        show_alert_if_fall(label, confidence)
                        log_prediction(label, confidence)

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            os.unlink(video_path)
            st.markdown(
                """
                <div class="sf-alert sf-alert-safe">
                  <div class="sf-alert-icon">✓</div>
                  <div><div class="sf-alert-title">Video processing complete</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# RIGHT COLUMN: monitoring dashboard
# ----------------------------------------------------------------------
with col_dashboard:
    st.markdown('<div class="sf-section-label">Monitoring analytics</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sf-metric-row">
          <div class="sf-metric">
            <div class="sf-metric-label">Total</div>
            <div class="sf-metric-value">{st.session_state.total_detected}</div>
          </div>
          <div class="sf-metric sf-metric-fall">
            <div class="sf-metric-label">Falls</div>
            <div class="sf-metric-value">{st.session_state.fall_count}</div>
          </div>
          <div class="sf-metric sf-metric-normal">
            <div class="sf-metric-label">Normal</div>
            <div class="sf-metric-value">{st.session_state.normal_count}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.activity_log:
        log_df = pd.DataFrame(st.session_state.activity_log)

        st.markdown('<div class="sf-section-label">Activity distribution</div>', unsafe_allow_html=True)
        st.bar_chart(log_df["activity"].value_counts(), color="#0E6E62")

        st.markdown('<div class="sf-section-label">Recent predictions</div>', unsafe_allow_html=True)
        st.dataframe(log_df.tail(10), width="stretch", hide_index=True)
    else:
        st.markdown(
            """
            <div class="sf-card">
              <span style="color: var(--muted); font-size: 0.9rem;">
                No activity logged yet. Upload an image or video to begin monitoring.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("---")
st.markdown(
    """
    <p style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted); letter-spacing:0.02em;">
      SAFEFALL AI PROTOTYPE &middot; FA-2 DELIVERABLE &middot; POSE ESTIMATION: MEDIAPIPE &middot;
      CLASSIFIER: 1D-CNN ON POSE LANDMARKS &middot; NOT FOR REAL CLINICAL USE
    </p>
    """,
    unsafe_allow_html=True,
)
