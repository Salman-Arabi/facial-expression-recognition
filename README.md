# Facial Expression Recognition

Real-time emotion detection from webcam using MediaPipe face detection + a TensorFlow Lite emotion classifier trained on AffectNet (400K+ real-world face images).

Detects 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise.

## Quick Demo

```bash
# 1. Create the inference environment
python -m venv fer_face_env
fer_face_env\Scripts\pip install opencv-python mediapipe ai-edge-litert numpy

# 2. Run the demo
fer_face_env\Scripts\python src\webcam_demo.py
```

Press `q` to quit.

## Project Structure

```
Facial-Expression-Recognition/
├── src/
│   ├── webcam_demo.py      Real-time webcam emotion detection
│   └── train_model.py       CNN training on FER2013 (optional)
├── models/
│   ├── emotiefflib_mobilenet_7.tflite   Primary model (AffectNet, 224x224)
│   ├── best_model.tflite                 Backup model (FER2013, 48x48)
│   └── face_detection_short_range.tflite MediaPipe face detection model
├── data/                    FER2013 dataset (train/ + test/)
├── requirements.txt         Dependency notes
└── README.md
```

## Model

The main model (`emotiefflib_mobilenet_7.tflite`) is a MobileNet-based classifier from the [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) research project. It was:

- Pre-trained on VGGFace2 for face identification
- Fine-tuned on **AffectNet** (400K+ manually annotated facial images)
- Achieves **64.71% accuracy** on the AffectNet 7-class benchmark
- Runs at ~16ms per inference on mobile-class CPUs

The model operates on **224x224 RGB** face crops with raw pixel input (no normalization).

## Setup

### Inference environment (for running the demo)

```bash
python -m venv fer_face_env
fer_face_env\Scripts\activate
pip install opencv-python mediapipe ai-edge-litert numpy
```

> **Note:** You must install `mediapipe` with `--no-deps` if `protobuf` conflicts arise, then install `ai-edge-litert` separately for the TFLite runtime.

### Training environment (optional, for retraining)

```bash
python -m venv fer_env
fer_env\Scripts\activate
pip install tensorflow matplotlib scikit-learn
```

Training and inference environments cannot share the same venv due to protobuf version conflicts between mediapipe and tensorflow.

## Training Your Own Model

The included `src/train_model.py` trains a CNN from scratch on the FER2013 dataset:

```bash
fer_env\Scripts\python src\train_model.py
```

The FER2013 dataset is included in `data/`. You can also download it from [Kaggle](https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013).

## How It Works

1. **Face detection**: MediaPipe's face detection model locates faces in each webcam frame
2. **Crop & resize**: Each detected face is cropped with a tight margin (−10% padding) and resized to 224x224
3. **Inference**: The TFLite model classifies the face region into one of 7 emotion categories
4. **Display**: Results are shown with bounding boxes, confidence scores, and a per-emotion probability distribution


