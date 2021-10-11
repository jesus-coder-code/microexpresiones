import cv2
import mediapipe as mp
import math

captura = cv2.VideoCapture(1)
captura.set(3, 1280)
captura.set(3, 720)

mpDibujar = mp.solutions.drawing_utils
ConfigDibujo = mpDibujar.DrawingSpec(thickness=1, circle_radius=1)

mpMalla = mp.solutions.face_mesh
Malla = mpMalla.FaceMesh(max_num_faces=1)

while True:
    ret, frame = captura.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = Malla.process(frameRGB)

    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if res.multi_face_landmarks:
        for rostro in res.multi_face_landmarks:
            mpDibujar.draw_landmarks(frame, rostro, mpMalla.FACE_CONTOURS, ConfigDibujo, ConfigDibujo)

            for id, puntos in enumerate(rostro.landmark):
                al, an, c = frame.shape
                x, y = int(puntos. x * an), int(puntos. y * al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])

                if len(lista) == 468:
                    #ceja derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    #cv2.line(frame, (x1, y1), (x2, y2), (0,0,0), t)
                    #cv2.line(frame, (x1, y1), r, (0,0,0), cv2,FILLED)
                    #cv2.line(frame, (x2, y2), r, (0,0,0), cv2,FILLED)
                    #cv2.line(frame, (cx, cy), r, (0,0,0), cv2,FILLED)
                    lon1 = math.hypot(x2 - x1, y2 - y1)

                    #ceja izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    lon2 = math.hypot(x4 - x3, y4 - y3)

                    #extremos boca
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    lon3 = math.hypot(x6 - x5, y6 - y5)

                    #boca abierta
                    x7, y7 = lista[78][1:]
                    x8, y8 = lista[308][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    lon4 = math.hypot(x8 - x7, y8 - y7)
    

    cv2.imshow('microexpresiones faciales', frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

captura.release()
cv2.destroyAllWindows()





