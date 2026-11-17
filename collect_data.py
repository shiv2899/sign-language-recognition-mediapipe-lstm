import cv2
import numpy as np
import os
import mediapipe as mp
from time import sleep

ACTIONS = np.array(['Hello', 'Yes', 'No'])
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
DATA_PATH = 'data'

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

for action in ACTIONS:
    for seq in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: break
                image, results = mediapipe_detection(frame, holistic)
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting {action} - Video {sequence+1}/{NO_SEQUENCES}', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, f'{action} - Video {sequence+1}/{NO_SEQUENCES} - Frame {frame_num}', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('Collect Data - Q dabake band karo', image)
                keypoints = extract_keypoints(results)
                np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)
                if cv2.waitKey(10) & 0xFF == ord('q'): break
            print(f"Done: {action} - Video {sequence+1}/{NO_SEQUENCES}")
            sleep(0.5)

cap.release()
cv2.destroyAllWindows()
print("Data Collection DONE!")