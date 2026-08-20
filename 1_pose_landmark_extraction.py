"""
STEP 4-5 (Part A): Pose Landmark Extraction
--------------------------------------------
Reads the activity-wise frame folders you already created in FA-1
(/fall, /walking, /sitting, /standing, /normal), runs MediaPipe Pose on
every frame, extracts 33 body keypoints (x, y, z, visibility = 132
features per frame), and saves everything into a single CSV that will
be used to train the classification model in Step 2 (2_train_model.py).

Run this in Google Colab or Jupyter Notebook.

Expected folder structure (created during FA-1 preprocessing):

dataset/
├── fall/
│   ├── frame001.jpg
│   ├── frame002.jpg
├── walking/
├── sitting/
├── standing/
└── normal/

Install dependencies first (Colab cell):
    !pip install mediapipe opencv-python pandas tqdm
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG - update DATASET_DIR to point to your FA-1 organized frames
# ----------------------------------------------------------------------
DATASET_DIR = "dataset"          # folder containing fall/, walking/, sitting/, standing/, normal/
OUTPUT_CSV = "landmarks_dataset.csv"
ACTIVITY_CLASSES = ["fall", "walking", "sitting", "standing", "normal"]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ----------------------------------------------------------------------
# MediaPipe Pose setup
# ----------------------------------------------------------------------
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
)


def extract_landmarks_from_image(image_path):
    """
    Runs MediaPipe Pose on a single image and returns a flat list of
    132 values: [x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32]
    Returns None if no pose was detected in the frame.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    row = []
    for landmark in results.pose_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return row


def build_dataset():
    all_rows = []
    skipped = 0

    for activity in ACTIVITY_CLASSES:
        folder_path = os.path.join(DATASET_DIR, activity)
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Folder not found, skipping: {folder_path}")
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        print(f"Processing '{activity}': {len(image_files)} frames")

        for file_name in tqdm(image_files, desc=activity):
            file_path = os.path.join(folder_path, file_name)
            landmarks = extract_landmarks_from_image(file_path)

            if landmarks is None:
                skipped += 1
                continue

            row = landmarks + [activity]
            all_rows.append(row)

    # Build column names: feature_0 ... feature_131, label
    feature_columns = [f"feature_{i}" for i in range(132)]
    columns = feature_columns + ["label"]

    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n----- Extraction Summary -----")
    print(f"Total usable frames extracted : {len(df)}")
    print(f"Frames skipped (no pose found): {skipped}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Saved landmark dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    build_dataset()
    pose_detector.close()
