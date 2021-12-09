import cv2
import math
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  # cv2_imshow(img)

# Read images with OpenCV.
# images = {name: cv2.imread(name) for name in uploaded.keys()}
# Preview the images.
name = '000001.png'
print(name)
image = cv2.imread(name)
# resize_and_show(image)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print(results.multi_face_landmarks)