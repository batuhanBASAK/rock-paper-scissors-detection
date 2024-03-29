import cv2
import mediapipe as mp
import numpy as np
import pickle

classes = ['paper', 'rock', 'scissors']

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


with open('model.pickle', 'rb') as handle:
    clf = pickle.load(handle)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    
    if results.multi_hand_landmarks is not None:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, 
                                        landmarks, 
                                        mp_hands.HAND_CONNECTIONS, 
                                        circles_color, 
                                        lines_color)

            X = []
            for landmark in landmarks.landmark:
                x = landmark.x
                y = landmark.y
                X.append(x)
                X.append(y)

            X = np.asarray(X)
            y_pred = np.argmax(clf.predict(X.reshape(1, -1)))
            print(classes[y_pred])


            h, w, c = frame.shape

            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            padding = 40
            cv2.rectangle(frame, (x_min-padding, y_min-padding), (x_max+padding, y_max+padding), (255, 0, 0), 5)

            frame = cv2.putText(frame, 
                classes[y_pred], 
                (x_min-padding, y_min-padding-10), 
                cv2.FONT_HERSHEY_SIMPLEX,
                2, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA)



    cv2.imshow('frame', frame)


    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
