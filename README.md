Canva Video Link: https://canva.link/mrh2upt6jwd2uxh
# SafeFall AI – Elderly Fall Detection System

## Project Overview

SafeFall AI is an AI-powered healthcare monitoring system designed to detect elderly falls and classify daily activities using Computer Vision, Pose Estimation, Machine Learning, and Deep Learning techniques.

The system analyzes human body posture and movement patterns to identify activities such as:

*  Walking
*  Sitting
*  Standing
*  Normal Activity
*  Fall Detection

Whenever a fall is detected, the system generates emergency alerts to assist caregivers, hospitals, and elderly care centers in responding quickly to potential accidents.

This project was developed as part of the FA-2 Machine Learning and Deep Learning Project, focusing on the practical implementation and deployment of an AI-powered healthcare monitoring solution.



##  Objectives

The primary objectives of this project are:

* Detect elderly falls using AI and Computer Vision.
* Classify human activities from image and video inputs.
* Improve patient safety through automated monitoring.
* Generate emergency alerts during fall incidents.
* Deploy a user-friendly healthcare dashboard using Streamlit.
* Demonstrate the application of Deep Learning in healthcare monitoring systems.


##  Problem Statement

Falls are one of the leading causes of injury among elderly individuals. Continuous monitoring by caregivers is often difficult, especially in homes, hospitals, and elderly care facilities.

SafeFall AI addresses this challenge by providing an automated monitoring system capable of:

* Detecting falls in real time.
* Monitoring daily activities.
* Reducing response time during emergencies.
* Supporting healthcare professionals with AI-assisted monitoring.

##  Technologies Used

### Programming Language

* Python

### Computer Vision

* OpenCV
* MediaPipe Pose Estimation

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit
* Streamlit Cloud

##  Dataset

The project utilizes the Le2i Fall Detection Dataset, which contains videos and frames representing various human activities.

### Activity Classes

* Fall
* Walking
* Sitting
* Standing
* Normal Activity

### Dataset Processing

The dataset underwent:

* Video extraction
* Frame generation
* Image resizing
* Data labeling
* Activity categorization
* Feature extraction

##  System Workflow

### Step 1: Data Collection

Acquire elderly activity videos and fall incident samples.

### Step 2: Data Preprocessing

* Extract frames from videos.
* Resize images.
* Normalize pixel values.
* Organize activity classes.

### Step 3: Pose Estimation

MediaPipe Pose is used to detect human body landmarks such as:

* Shoulders
* Elbows
* Hips
* Knees
* Ankles

### Step 4: Feature Extraction

Pose landmarks are converted into meaningful numerical features representing body posture and movement.

### Step 5: Activity Classification

A Deep Learning model is trained to classify activities into:

* Fall
* Walking
* Sitting
* Standing
* Normal Activity

### Step 6: Alert Generation

When a fall is detected:

* Warning messages are displayed.
* Emergency alerts are generated.

### Step 7: Dashboard Deployment

The trained model is deployed through Streamlit for user interaction.

## Model Architecture

The AI pipeline consists of:

Input Video/Image

⬇

Pose Estimation (MediaPipe)

⬇

Feature Extraction

⬇

Deep Learning Classification Model

⬇

Activity Prediction

⬇

Alert System

⬇

Streamlit Dashboard

## Model Evaluation

The model was evaluated using:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures how many predicted falls were actually falls.

### Recall

Measures how many actual falls were successfully detected.

### F1-Score

Balances Precision and Recall.

### Confusion Matrix

Used to analyze:

* Correct predictions
* False alarms
* Missed fall detections
* Misclassified activities

##  Alert System

The alert system is designed to improve emergency response.

### Features

* Fall Detection Alerts
* Emergency Warning Messages
* Visual Notifications
* Monitoring Dashboard Alerts

Example:

WARNING: Fall Detected!

Immediate attention may be required.

##  Streamlit Dashboard Features

The deployed web application provides:

### Image Upload

Users can upload images for activity prediction.

### Video Upload

Users can upload videos for activity analysis.

### AI Prediction

The model predicts the detected activity.

### Pose Visualization

Displays detected body landmarks.

### Monitoring Analytics

Shows system statistics and predictions.

### Alert Generation

Triggers notifications during fall events.

## Live Application

### Streamlit Deployment

SafeFall AI Web App:

https://harinipriya1000410-safe-fall-ai.streamlit.app/

## Future Improvements

The system can be enhanced by:

* Supporting live CCTV monitoring.
* Improving performance in low-light environments.
* Adding more training data.
* Increasing pose estimation accuracy.
* Reducing false alarms.
* Implementing real-time notifications.
* Integrating SMS and email alerts.
* Supporting multiple camera feeds.
* Developing mobile healthcare monitoring applications.


## Challenges Faced

During development, several challenges were encountered:

* Lighting variations.
* Camera angle differences.
* Occlusion of body parts.
* Similarity between sitting and falling postures.
* Limited fall activity samples.
* Real-time processing constraints.

These challenges provide opportunities for future improvements and research.



## Learning Outcomes

Through this project, the following skills were developed:

* Computer Vision
* Pose Estimation
* Deep Learning Model Training
* Activity Classification
* Healthcare AI Applications
* Model Evaluation
* Streamlit Deployment
* Real-Time Monitoring Systems



© 2026 Harini Priya Karthikeyan. All Rights Reserved.
