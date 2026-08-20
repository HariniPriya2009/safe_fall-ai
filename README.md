# SafeFall AI — Elderly Fall Detection System (FA-2)

CareVision HealthTech Pvt. Ltd. | AI-Powered Elderly Fall Detection using MediaPipe Pose + 1D-CNN, deployed via Streamlit.

This repo covers **Steps 4–7** of FA-2, continuing from the FA-1 storyboard (problem definition, preprocessing, EDA).

## Pipeline overview

| File | Step | What it does |
|---|---|---|
| `1_pose_landmark_extraction.py` | Step 4–5 (feature extraction) | Runs MediaPipe Pose on your FA-1 frame folders, saves 132 landmark features per frame to `landmarks_dataset.csv` |
| `2_train_model.py` | Step 5 (training) | Splits data 70/15/15, trains a 1D-CNN classifier, saves model + accuracy/loss plots |
| `3_evaluate_model.py` | Step 6 (evaluation) | Computes Accuracy/Precision/Recall/F1, confusion matrix |
| `app.py` | Step 7 (deployment) | Streamlit dashboard: upload image/video, get predictions, fall alerts, monitoring analytics |

## 1. Run in Google Colab / Jupyter

```bash
!pip install mediapipe opencv-python pandas tqdm tensorflow scikit-learn matplotlib seaborn joblib
```

Upload your FA-1 organized dataset (`dataset/fall`, `dataset/walking`, `dataset/sitting`,
`dataset/standing`, `dataset/normal`), then run in order:

```bash
python 1_pose_landmark_extraction.py
python 2_train_model.py
python 3_evaluate_model.py
```

This produces the evidence files FA-2 asks for:
- `landmarks_dataset.csv`
- `accuracy_plot.png`, `loss_plot.png`
- `confusion_matrix.png`, `classification_report.txt`
- `fall_detection_model.h5`, `feature_scaler.pkl`, `label_encoder_classes.npy`

Take screenshots of pose detection output, training curves, and the confusion matrix for your
recorded video walkthrough.

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "FA-2: fall detection pipeline + Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**Important:** commit the trained artifacts too (`fall_detection_model.h5`, `feature_scaler.pkl`,
`label_encoder_classes.npy`) — `app.py` loads them directly from the repo root. Do **not** commit
the raw `dataset/` folder (too large); only the code + trained artifacts are needed for deployment.

## 3. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, select your repo, branch `main`, and set **Main file path** to `app.py`.
3. Deploy. Streamlit Cloud will install everything from `requirements.txt` automatically.
4. Copy the live app URL — this is your "Link to deployed project" for the FA-2 checklist.

## 4. Test locally (optional, before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes on model design choice

The classifier operates on **MediaPipe pose landmarks** (132 features: x, y, z, visibility for
33 body keypoints) rather than raw pixels. This directly follows the FA-2 workflow — "Detecting
body landmarks using pose estimation → Extracting features from posture and movement → Training
the deep learning classification model" — and is lighter weight and more robust to background/
lighting variation than a pure image CNN, while still using a CNN architecture (1D-CNN over the
landmark vector) as recommended in Step 4.

## Step 8: Monitoring & Maintenance (write-up, not code)

Include in your final report/video:
- Periodic retraining plan using new Le2i-style recordings or CareVision's own facility footage
- Plan to reduce false alerts (raise `ALERT_CONFIDENCE_THRESHOLD` in `app.py`, collect more
  "sitting vs falling" edge-case samples)
- Handling low-light: augment training data with brightness-adjusted frames (from FA-1 EDA)
- Path to real-time CCTV: swap the video-upload loop in `app.py` for a live `cv2.VideoCapture(0)`
  or an RTSP stream once deployed on-premise instead of Streamlit Cloud
