'''
detección de microexpresiones
'''

#USAGE : python test.py

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import mediapipe as mp


face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['ceno fruncido','sonrisa','Neutral','boca abajo','cejas arriba-boca abierta']
# class_labels = ['Enojo','Felicidad','Neutral','Tristeza','Sorpresa']

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(3, 720)

mpDibujar = mp.solutions.drawing_utils
ConfigDibujo = mpDibujar.DrawingSpec(thickness=1, circle_radius=1)

mpMalla = mp.solutions.face_mesh
Malla = mpMalla.FaceMesh(max_num_faces=1)
px = []
py = []
lista = []

while True:
    # toma de video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = Malla.process(frameRGB)

    if res.multi_face_landmarks:
        for rostro in res.multi_face_landmarks:
            mpDibujar.draw_landmarks(frame, rostro, mpMalla.FACEMESH_CONTOURS, ConfigDibujo, ConfigDibujo)

            for id, puntos in enumerate(rostro.landmark):
                al, an, c = frame.shape
                x, y = int(puntos.x*an), int(puntos.y*al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # predicción de los resultados y mostrar estos en pantalla

            preds = classifier.predict(roi)[0]
            #print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            #print("\nprediction max = ",preds.argmax())
            #print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.8,(15,22,238),3)
        else:
            cv2.putText(frame,'No se encuentra un rostro',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(15,22,238),3)
        print("\n\n")
    cv2.imshow('Deteccion de microexpresiones',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
