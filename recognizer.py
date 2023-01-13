import cv2
import mediapipe as mp
from mediapipe import * 

upFal = [4, 8, 12, 16, 20]
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode = False,
                                 max_num_hands = 1,
                                 min_tracking_confidence = 0.5,
                                 min_detection_confidence = 0.5)

mpDraw = mp.solutions.drawing_utils

while True: 
    _, img = cap.read()
    img = cv2.flip(img, 1)
    
    result = hands.process(img)
    
    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0))
            
            if id in upFal:
                cv2.circle(img, (cx, cy), 20, (255, 255, 0))

            
                        
        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        
    cv2.imshow("Frame", img)
    cv2.waitKey(1)