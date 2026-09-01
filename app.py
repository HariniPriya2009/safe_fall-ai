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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    /* Pure black canvas */
    --bg-0: #000000;
    --bg-1: #03000A;
    --bg-2: #07000F;
    /* Glass surfaces — near-black, purple-tinted */
    --glass: rgba(12, 6, 24, 0.55);
    --glass-light: rgba(20, 10, 38, 0.40);
    --glass-border: rgba(176, 38, 255, 0.14);
    --glass-border-strong: rgba(176, 38, 255, 0.30);
    /* Text — crisp white */
    --ink: #FFFFFF;
    --ink-soft: #E8E8F5;
    --muted: #8A8AA8;
    /* Neon purple hero accent + companions (all in the purple/magenta family) */
    --purple: #B026FF;          /* electric violet — hero */
    --purple-deep: #7B2CBF;     /* deep purple — gradient partner */
    --purple-dim: rgba(176, 38, 255, 0.14);
    --magenta: #FF2E97;         /* hot magenta — emergency / fall */
    --magenta-dim: rgba(255, 46, 151, 0.12);
    --magenta-glow: rgba(255, 46, 151, 0.50);
    --violet: #9D4EDD;          /* soft violet — safe / normal */
    --violet-dim: rgba(157, 78, 221, 0.12);
    --violet-glow: rgba(157, 78, 221, 0.40);
    --line: rgba(176, 38, 255, 0.12);
}

/* ===================== GLOBAL TYPE & BASE ===================== */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    color: var(--ink);
}

/* Pure black base with faint neon-purple radial glows (classy cyber-luxe depth) */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 700px at 10% -10%, rgba(176, 38, 255, 0.16), transparent 60%),
        radial-gradient(1000px 800px at 108% 6%, rgba(255, 46, 151, 0.10), transparent 55%),
        radial-gradient(900px 900px at 50% 125%, rgba(123, 44, 191, 0.10), transparent 60%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    background-attachment: fixed;
}

[data-testid="stMain"] { background: transparent; }

.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1280px; }

/* ===================== GLASS PRIMITIVE ===================== */
.sf-glass {
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.55),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.sf-glass-tight { border-radius: 14px; }

/* ===================== HEADER ===================== */
.sf-header { padding: 0.4rem 0 1rem 0; }
.sf-eyebrow {
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--purple); margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(176, 38, 255, 0.65);
}
.sf-title {
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 3rem;
    line-height: 1.05; letter-spacing: -0.02em; color: var(--ink); margin: 0;
    background: linear-gradient(180deg, #FFFFFF 0%, #E0B0FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sf-subtitle {
    font-size: 1.04rem; font-weight: 400; color: var(--ink-soft); margin-top: 0.7rem;
    max-width: 660px; line-height: 1.55;
}

/* Glowing ECG pulse divider — neon purple gradient */
.sf-pulse { width: 100%; height: 40px; margin: 1.3rem 0 0.5rem 0; display: block;
    filter: drop-shadow(0 0 10px rgba(176, 38, 255, 0.75)); }

/* ===================== SIDEBAR ===================== */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(10, 5, 20, 0.88), rgba(3, 0, 10, 0.88));
    backdrop-filter: blur(24px) saturate(130%);
    -webkit-backdrop-filter: blur(24px) saturate(130%);
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"] * { color: var(--ink-soft) !important; }
[data-testid="stSidebar"] .sf-sidebar-title, [data-testid="stSidebar"] .sf-sidebar-sub { color: var(--ink) !important; }

.sf-badge {
    width: 46px; height: 46px; border-radius: 13px;
    background: linear-gradient(145deg, rgba(176,38,255,0.35), rgba(255,46,151,0.18));
    border: 1px solid var(--glass-border-strong);
    display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 0 26px rgba(176, 38, 255, 0.45), inset 0 1px 0 rgba(255,255,255,0.08);
    animation: sf-badge-glow 3.4s ease-in-out infinite;
}
@keyframes sf-badge-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(176, 38, 255, 0.38), inset 0 1px 0 rgba(255,255,255,0.08); }
    50%      { box-shadow: 0 0 38px rgba(176, 38, 255, 0.65), inset 0 1px 0 rgba(255,255,255,0.08); }
}
.sf-sidebar-title { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.28rem; margin: 0; letter-spacing: -0.01em; }
.sf-sidebar-sub { font-size: 0.82rem; color: var(--muted); margin-top: 0.1rem; }

/* ===================== SECTION LABEL ===================== */
.sf-section-label {
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.7rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.7rem;
}

/* ===================== METRIC CARDS ===================== */
.sf-metric-row { display: flex; gap: 0.8rem; }
.sf-metric {
    flex: 1; padding: 1rem 1.1rem; border-radius: 16px;
    background: var(--glass);
    backdrop-filter: blur(18px) saturate(130%);
    -webkit-backdrop-filter: blur(18px) saturate(130%);
    border: 1px solid var(--glass-border);
    box-shadow: 0 6px 24px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.sf-metric:hover { transform: translateY(-2px); border-color: var(--glass-border-strong);
    box-shadow: 0 0 26px rgba(176, 38, 255, 0.25), inset 0 1px 0 rgba(255,255,255,0.04); }
.sf-metric-label {
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.66rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
}
.sf-metric-value {
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 2rem;
    color: var(--ink); margin-top: 0.2rem; letter-spacing: -0.02em;
}
.sf-metric-fall .sf-metric-value { color: var(--magenta); text-shadow: 0 0 22px var(--magenta-glow); }
.sf-metric-normal .sf-metric-value { color: var(--violet); text-shadow: 0 0 22px var(--violet-glow); }
.sf-metric-fall { border-color: rgba(255, 46, 151, 0.24); }
.sf-metric-normal { border-color: rgba(157, 78, 221, 0.22); }

/* ===================== ALERT BANNERS ===================== */
.sf-alert {
    border-radius: 16px; padding: 1.05rem 1.25rem; display: flex; gap: 0.9rem;
    align-items: flex-start; margin: 0.7rem 0;
    backdrop-filter: blur(16px) saturate(130%);
    -webkit-backdrop-filter: blur(16px) saturate(130%);
    animation: sf-alert-in 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes sf-alert-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.sf-alert-danger {
    background: var(--magenta-dim);
    border: 1px solid rgba(255, 46, 151, 0.38);
    box-shadow: 0 0 30px rgba(255, 46, 151, 0.28), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: sf-alert-in 0.5s cubic-bezier(0.22, 1, 0.36, 1), sf-danger-pulse 2.2s ease-in-out infinite 0.5s;
}
@keyframes sf-danger-pulse {
    0%, 100% { box-shadow: 0 0 22px rgba(255, 46, 151, 0.22), inset 0 1px 0 rgba(255,255,255,0.04); }
    50%      { box-shadow: 0 0 44px rgba(255, 46, 151, 0.50), inset 0 1px 0 rgba(255,255,255,0.04); }
}
.sf-alert-safe {
    background: var(--violet-dim);
    border: 1px solid rgba(157, 78, 221, 0.32);
    box-shadow: 0 0 26px rgba(157, 78, 221, 0.22), inset 0 1px 0 rgba(255,255,255,0.04);
}
.sf-alert-icon { font-size: 1.35rem; line-height: 1.4; }
.sf-alert-title { font-weight: 600; font-size: 0.98rem; letter-spacing: -0.01em; }
.sf-alert-danger .sf-alert-title { color: #FF6BB5; }
.sf-alert-safe .sf-alert-title { color: #B98CFF; }
.sf-alert-body { font-size: 0.88rem; color: var(--ink-soft); margin-top: 0.2rem; }

/* ===================== PREDICTION RESULT BANNER ===================== */
.sf-result {
    display: flex; align-items: baseline; gap: 0.7rem; padding: 0.95rem 1.2rem;
    background: linear-gradient(135deg, rgba(176,38,255,0.18), rgba(255,46,151,0.08));
    border: 1px solid var(--glass-border-strong);
    border-radius: 14px; margin: 0.8rem 0;
    backdrop-filter: blur(16px) saturate(130%);
    -webkit-backdrop-filter: blur(16px) saturate(130%);
    box-shadow: 0 0 30px rgba(176, 38, 255, 0.22), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: sf-alert-in 0.45s cubic-bezier(0.22, 1, 0.36, 1);
}
.sf-result-label {
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.14em; color: var(--purple);
    text-shadow: 0 0 14px rgba(176, 38, 255, 0.50);
}
.sf-result-value {
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.35rem;
    color: var(--ink); letter-spacing: -0.01em;
}
.sf-result-conf {
    font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.95rem;
    color: var(--ink-soft); margin-left: auto;
}

/* ===================== EMPTY STATE CARD ===================== */
.sf-card {
    background: var(--glass);
    backdrop-filter: blur(18px) saturate(130%);
    -webkit-backdrop-filter: blur(18px) saturate(130%);
    border: 1px solid var(--glass-border); border-radius: 16px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ===================== STREAMLIT WIDGET OVERRIDES ===================== */
/* Buttons */
.stButton>button {
    border-radius: 12px;
    font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: 0.01em;
    background: linear-gradient(145deg, rgba(176,38,255,0.26), rgba(255,46,151,0.14));
    border: 1px solid var(--glass-border-strong);
    color: var(--ink) !important;
    box-shadow: 0 0 22px rgba(176, 38, 255, 0.25), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: all 0.25s ease;
}
.stButton>button:hover {
    background: linear-gradient(145deg, rgba(176,38,255,0.40), rgba(255,46,151,0.22));
    box-shadow: 0 0 34px rgba(176, 38, 255, 0.45), inset 0 1px 0 rgba(255,255,255,0.08);
    transform: translateY(-1px);
}

/* Radio */
[data-testid="stRadio"] > div { gap: 0.4rem; }
[data-testid="stRadio"] label {
    background: var(--glass-light); border: 1px solid var(--glass-border);
    border-radius: 10px; padding: 0.5rem 0.8rem; margin-bottom: 0.35rem;
    transition: all 0.2s ease; cursor: pointer;
}
[data-testid="stRadio"] label:hover { border-color: var(--glass-border-strong); }
[data-testid="stRadio"] label[data-checked="true"],
[data-testid="stRadio"] label:has(svg) {
    border-color: var(--glass-border-strong);
    box-shadow: 0 0 18px rgba(176, 38, 255, 0.30);
}

/* ===== FILE UPLOADER — full dark-glass restyle (no white box) ===== */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] {
    background: var(--glass) !important;
    backdrop-filter: blur(18px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
    border: 1px dashed var(--glass-border-strong) !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--purple) !important;
    box-shadow: 0 0 30px rgba(176, 38, 255, 0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileDropzoneInstructions"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] * {
    color: var(--ink-soft) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileDropzoneInstructions"] > div > span {
    color: var(--ink) !important; font-weight: 500;
}
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileDropzoneInstructions"] small {
    color: var(--muted) !important; letter-spacing: 0.04em;
}

/* "Browse files" secondary button — neon purple glass */
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"],
section[data-testid="stFileUploader"] button[kind="secondary"],
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(145deg, rgba(176,38,255,0.24), rgba(255,46,151,0.12)) !important;
    border: 1px solid var(--glass-border-strong) !important;
    color: var(--ink) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 0 18px rgba(176, 38, 255, 0.22), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover,
section[data-testid="stFileUploader"] button[kind="secondary"]:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(145deg, rgba(176,38,255,0.38), rgba(255,46,151,0.20)) !important;
    box-shadow: 0 0 28px rgba(176, 38, 255, 0.40), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transform: translateY(-1px);
}

/* Uploaded file row — dark glass */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] ,
[data-testid="stFileUploader"] [data-baseweb="file-uploader"] [data-baseweb="block"],
[data-testid="stFileUploader"] ul,
[data-testid="stFileUploader"] li {
    background: var(--glass-light) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--ink-soft) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * ,
[data-testid="stFileUploader"] [data-baseweb="file-uploader"] * {
    color: var(--ink-soft) !important;
}
/* "Uploaded" status chip — violet glass pill */
[data-testid="stFileUploader"] [data-baseweb="tag"],
[data-testid="stFileUploader"] [role="status"] {
    background: rgba(157, 78, 221, 0.20) !important;
    border: 1px solid rgba(157, 78, 221, 0.40) !important;
    color: #B98CFF !important;
    border-radius: 999px !important;
}

/* Progress bar (video processing) — neon purple → magenta */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--purple), var(--magenta)) !important;
    box-shadow: 0 0 18px rgba(176, 38, 255, 0.60);
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--glass);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border); border-radius: 14px;
    overflow: hidden;
}
[data-testid="stDataFrame"] * { color: var(--ink-soft) !important; }
[data-testid="stDataFrame"] thead th { color: var(--purple) !important; font-weight: 600;
    text-shadow: 0 0 12px rgba(176, 38, 255, 0.45); }

/* Altair / bar chart */
.vega-embed, [data-testid="stVegaLiteChart"] {
    background: var(--glass) !important;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border); border-radius: 16px;
    padding: 0.8rem;
}

/* Image container — neon purple frame glow */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--glass-border);
    box-shadow: 0 8px 30px rgba(0,0,0,0.55), 0 0 20px rgba(176, 38, 255, 0.18);
}

/* Captions */
[data-testid="stImage"] caption, .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important; font-family: 'Inter', sans-serif; font-size: 0.8rem;
}

/* Dividers */
hr, [data-testid="stHorizontalBlock"] > hr {
    border-color: var(--line) !important; margin: 1rem 0;
}
[data-testid="stSidebar"] hr { border-color: var(--glass-border) !important; }

/* Error message */
[data-testid="stAlert"] {
    background: var(--magenta-dim) !important;
    border: 1px solid rgba(255, 46, 151, 0.38) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
[data-testid="stAlert"] * { color: #FFB8D8 !important; }

/* Footer */
.sf-footer {
    font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.72rem;
    color: var(--muted); letter-spacing: 0.08em; text-align: center; padding: 0.5rem 0;
}
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
VIDEO_FRAME_SAMPLE_RATE = 5 
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


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


if "total_detected" not in st.session_state:
    st.session_state.total_detected = 0
if "fall_count" not in st.session_state:
    st.session_state.fall_count = 0
if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []  # list of dicts: {activity, confidence}

def draw_pose_landmarks(image_bgr, landmarks):
    h, w = image_bgr.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(image_bgr, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)

    return image_bgr


def predict_frame(frame_bgr, model, scaler, class_names, pose_detector):
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

PULSE_SVG = """
<svg class="sf-pulse" viewBox="0 0 600 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="pulseGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#7B2CBF" stop-opacity="0.2"/>
      <stop offset="40%" stop-color="#B026FF" stop-opacity="0.95"/>
      <stop offset="60%" stop-color="#FF2E97" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#7B2CBF" stop-opacity="0.2"/>
    </linearGradient>
  </defs>
  <polyline points="0,20 140,20 160,20 172,4 184,36 196,8 208,20 240,20 600,20"
    fill="none" stroke="url(#pulseGrad)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
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
        st.bar_chart(log_df["activity"].value_counts(), color="#B026FF")

        st.markdown('<div class="sf-section-label">Recent predictions</div>', unsafe_allow_html=True)
        st.dataframe(log_df.tail(10), width="stretch", hide_index=True)
    else:
        st.markdown(
            """
            <div class="sf-card">
              <span style="color: var(--ink-soft); font-size: 0.9rem;">
                No activity logged yet. Upload an image or video to begin monitoring.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("---")
st.markdown(
    """
    <p class="sf-footer">
      SAFEFALL AI PROTOTYPE &middot; FA-2 DELIVERABLE &middot; POSE ESTIMATION: MEDIAPIPE &middot;
      CLASSIFIER: 1D-CNN ON POSE LANDMARKS &middot; NOT FOR REAL CLINICAL USE
    </p>
    """,
    unsafe_allow_html=True,
)
