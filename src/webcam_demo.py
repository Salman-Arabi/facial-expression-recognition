import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "emotiefflib_mobilenet_7.tflite")
IMG_SIZE = (224, 224)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😊",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

EMOTION_COLORS = {
    "angry": (0, 0, 200),
    "disgust": (0, 180, 0),
    "fear": (180, 0, 180),
    "happy": (0, 220, 220),
    "neutral": (140, 140, 140),
    "sad": (200, 100, 0),
    "surprise": (220, 220, 0),
}

CROP_PADDING = -0.1
FACE_DETECTION_CONFIDENCE = 0.5
MIN_CONFIDENCE = 0.2
SHOW_DISTRIBUTION = True

DEBUG_SAVE_CROP = False

print("Loading TFLite model ...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_idx = input_details[0]["index"]
output_idx = output_details[0]["index"]
print(f"Model input:  {input_details[0]['shape']}  {input_details[0]['dtype']}")
print(f"Model output: {output_details[0]['shape']}  {output_details[0]['dtype']}")
print(f"Emotions: {EMOTIONS}")

print("Initializing MediaPipe Face Detection ...")
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=FACE_DETECTION_CONFIDENCE,
)

print("Opening webcam (press 'q' to quit) ...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {frame_width}x{frame_height}")


def preprocess_face(face_roi):
    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)


def draw_text_with_bg(frame, text, x, y, font_scale=0.55, thickness=2, padding=5, fg=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale,
                fg, thickness, cv2.LINE_AA)


def draw_distribution(frame, predictions, x, y, bar_width=180, bar_height=14, gap=2):
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        bx = x
        by = y + i * (bar_height + gap)
        color = EMOTION_COLORS[emotion]
        filled = int(bar_width * min(prob, 1.0))

        cv2.rectangle(frame, (bx, by), (bx + bar_width, by + bar_height),
                      (40, 40, 40), -1)
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + bar_height),
                          color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_width, by + bar_height),
                      (80, 80, 80), 1)

        label = f"{emotion} {prob*100:.0f}%"
        label_x = bx + bar_width + 6
        label_y = by + bar_height - 3
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


print("\n=== Emotion Detection Demo Running ===")
print("Press 'q' in the video window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            cx = int(bbox.xmin * w) + int(bbox.width * w) // 2
            cy = int(bbox.ymin * h) + int(bbox.height * h) // 2
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            crop_w = int(bw * (1 + 2 * CROP_PADDING))
            crop_h = int(bh * (1 + 2 * CROP_PADDING))
            crop_w = max(crop_w, 10)
            crop_h = max(crop_h, 10)

            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(w, cx + crop_w // 2)
            y2 = min(h, cy + crop_h // 2)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            if DEBUG_SAVE_CROP:
                cv2.imwrite("debug_crop.png", face_roi)

            input_tensor = preprocess_face(face_roi)
            interpreter.set_tensor(input_idx, input_tensor)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_idx)[0]

            emotion_idx = int(np.argmax(predictions))
            confidence = float(predictions[emotion_idx])

            if confidence >= MIN_CONFIDENCE:
                emoji = EMOJI_MAP.get(EMOTIONS[emotion_idx], "")
                label = f"{EMOTIONS[emotion_idx]} {confidence * 100:.1f}% {emoji}"
            else:
                label = f"... {predictions[emotion_idx]*100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_text_with_bg(frame, label, x1, max(y1 - 10, 5))

            if SHOW_DISTRIBUTION:
                dist_x = min(x1, w - 240)
                dist_y = y2 + 15
                if dist_y + 7 * 16 > h:
                    dist_y = max(y1 - 7 * 16 - 10, 5)
                draw_distribution(frame, predictions, dist_x, dist_y)

    cv2.imshow("Emotion Detection Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()
print("Demo exited cleanly.")
