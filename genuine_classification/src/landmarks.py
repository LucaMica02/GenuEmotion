import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

gray_image = cv2.imread('resized_face.png', cv2.IMREAD_GRAYSCALE)
height, width = gray_image.shape
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

results = face_mesh.process(rgb_image)
for facial_landmarks in results.multi_face_landmarks:
    for pt in facial_landmarks.landmark:
        x = int(pt.x * width)
        y = int(pt.y * height)
        cv2.circle(rgb_image, (x, y), 2, (0, 255, 0), -1)

cv2.imshow('Face Mesh', rgb_image)
cv2.waitKey(0)