import cv2
import numpy as np
import dlib

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
     _, img = video_capture.read()
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     faces = detector(gray)
     for face in faces:
         x1 = face.left()
         y1 = face.top()
         x2 = face.right()
         y2 = face.bottom()

         landmarks = predictor(gray, face)
         for punto in range(0, 68):
             x = landmarks.part(punto).x
             y = landmarks.part(punto).y
             cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

     cv2.imshow("Marcas Faciales", img)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break
video_capture.release()
cv2.destroyAllWindows()
