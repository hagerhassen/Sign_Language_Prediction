import joblib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import os
import argparse
import warnings
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

script_parser = argparse.ArgumentParser(description="options to run the inference model")
script_parser.add_argument('--input', '-i', type=str, help="0 for camera or path for video")
args = script_parser.parse_args()

model = load_model('model.h5')
encoder = joblib.load('encoder.pkl')
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    la = np.array([[res.x, res.y, res.z] if res.visibility > 0.2 else [0, 0, 0] for res in
                   np.array(results.pose_landmarks.landmark)[[13, 15]]]) if results.pose_landmarks else np.zeros((2, 3))
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    ra = np.array([[res.x, res.y, res.z] if res.visibility > 0.2 else [0, 0, 0] for res in
                   np.array(results.pose_landmarks.landmark)[[14, 16]]]) if results.pose_landmarks else np.zeros((2, 3))
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([la, lh, ra, rh])


captureLocation = args.input if args.input != "0" else 0
capture = cv2.VideoCapture(captureLocation)
currentPrediction = None
with mp_holistic.Holistic(min_detection_confidence=0.001, min_tracking_confidence=0.001) as holistic:
    results_arr = []
    while True:
        success, frame = capture.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = frame.copy()
            results_arr.append(extract_keypoints(results))
            if len(results_arr) >= 30:
                prediction = model.predict(np.array(results_arr[:30]).reshape((1, 30, 46, 3)))
                prediction_prob = prediction[0][np.argmax(prediction)] * 100
                if prediction_prob > 60:
                    currentPrediction = encoder.inverse_transform(np.array(np.argmax(prediction)).reshape(1, ))[0]
                del results_arr[:1]

            if currentPrediction:
                text_to_be_reshaped = currentPrediction
                reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 40)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10), bidi_text, font=font,fill="blue")
                image = np.array(img_pil)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imshow("test", image)

        if cv2.waitKey(70) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
