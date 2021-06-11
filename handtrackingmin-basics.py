import cv2
import mediapipe as mp

import time


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

prev_time = 0
current_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                print(id, lm)
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)

                if id == 0:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70,), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)


# TODO min 49.22




