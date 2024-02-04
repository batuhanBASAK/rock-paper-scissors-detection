import cv2
import os
import mediapipe as mp

import numpy as np
import pickle


classes = ['rock', 'scissors', 'paper']

IMG_SIZE = 100

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("camera couldn't been opened")
    exit(1)


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=.5,
                        min_tracking_confidence=.5,
                        model_complexity=0)

mp_drawing = mp.solutions.drawing_utils

circles_color = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=4)
lines_color = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5)


X = []
labels = []

for c in classes:
    for i in range(IMG_SIZE):
        text = 'class: ' +  c + ', image:' + str(i)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            key = cv2.waitKey(1) & 0xff
            if key == ord(' '):
                if results.multi_hand_landmarks is not None:
                    for landmarks in results.multi_hand_landmarks:
                        aux = []
                        for landmark in landmarks.landmark:
                            x = landmark.x
                            y = landmark.y
                            aux.append(x)
                            aux.append(y)
                        X.append(aux)
                        labels.append(c)
                break

            if results.multi_hand_landmarks is not None:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, 
                                              landmarks, 
                                              mp_hands.HAND_CONNECTIONS, 
                                              circles_color, 
                                              lines_color)


            frame = cv2.putText(frame, 
                                text, 
                                (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, 
                                (0, 255, 0), 
                                2, 
                                cv2.LINE_AA)

            cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()


X = np.asarray(X)
y = np.asarray(labels)

with open('dataset.pickle', 'wb') as handle:
    pickle.dump({'X': X, 'y': y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
