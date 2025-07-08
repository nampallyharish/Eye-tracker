import os
import cv2
import dlib
import pyautogui
import numpy as np
from imutils import face_utils

# Check if the shape predictor file exists
path = r'C:\Users\DELL\OneDrive\Desktop\EYE detect\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(path)
if not os.path.isfile(path):
    print("File not found at:", path)
    exit()
else:
    print("File found successfully!")

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

# Start webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

def midpoint(p1, p2):
    return int((p1.x + p2.x) // 2), int((p1.y + p2.y) // 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Get left and right eye landmarks
        left_eye_pts = []
        right_eye_pts = []

        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye_pts.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye_pts.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Find eye centers
        left_center = np.mean(left_eye_pts, axis=0).astype(int)
        right_center = np.mean(right_eye_pts, axis=0).astype(int)
        
        eye_center_x = (left_center[0] + right_center[0]) // 2
        eye_center_y = (left_center[1] + right_center[1]) // 2

        # Move mouse (mapping face region to screen size roughly)
        rel_x = eye_center_x / frame.shape[1]
        rel_y = eye_center_y / frame.shape[0]

        cursor_x = int(screen_width * rel_x)
        cursor_y = int(screen_height * rel_y)

        pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)
    
    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
