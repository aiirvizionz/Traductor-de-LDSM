import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from pathlib import Path
import time

# 1. Función para dibujar la malla de la mano
def dibujar_mano(imagen, hand_landmarks):
    H, W, _ = imagen.shape
    conexiones = [(0,1),(1,2),(2,3),(3,4),
                  (0,5),(5,6),(6,7),(7,8),
                  (5,9),(9,10),(10,11),(11,12),
                  (9,13),(13,14),(14,15),(15,16),
                  (13,17),(17,18),(18,19),(19,20),(0,17)]
    
    for inicio, fin in conexiones:
        x1, y1 = int(hand_landmarks[inicio].x * W), int(hand_landmarks[inicio].y * H)
        x2, y2 = int(hand_landmarks[fin].x * W), int(hand_landmarks[fin].y * H)
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    for lm in hand_landmarks:
        x, y = int(lm.x * W), int(lm.y * H)
        cv2.circle(imagen, (x, y), 5, (0, 0, 255), -1)

# 2. Configuración de MediaPipe Tasks
ruta_base = Path(__file__).parent
MODELO_LOCAL = ruta_base / 'DATASET' / 'hand_landmarker.task'

if not MODELO_LOCAL.exists():
    print(f"Error: No se encuentra el archivo del modelo en {MODELO_LOCAL}")
    exit()

opciones = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODELO_LOCAL)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

datos_extraidos = []

# Excluimos: CH, G, J, LL, Ñ, Q, RR, X, Z (señas dinámicas según tu imagen)
letras_a_capturar = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
muestras_por_letra = 200 # Aumentamos para mayor precisión

cap = cv2.VideoCapture(0)
print("Iniciando cámara...")

with vision.HandLandmarker.create_from_options(opciones) as landmarker:
    for letra in letras_a_capturar:
        muestras_tomadas = 0
        grabando = False
        
        while cap.isOpened() and muestras_tomadas < muestras_por_letra:
            exito, imagen = cap.read()
            if not exito:
                continue

            imagen = cv2.flip(imagen, 1)
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            imagen_mp_lista = np.ascontiguousarray(imagen_rgb)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_mp_lista)
            
            resultados = landmarker.detect(imagen_mp)

            # Interfaz visual
            if not grabando:
                cv2.putText(imagen, f"Haz la sena '{letra}' y presiona '{letra.lower()}' para grabar", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(imagen, f"GRABANDO '{letra}': Mueve tu mano ligeramente... {muestras_tomadas}/{muestras_por_letra}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if resultados.hand_landmarks:
                for hand_landmarks in resultados.hand_landmarks:
                    dibujar_mano(imagen, hand_landmarks)
                    
                    if grabando:
                        fila_coordenadas = []
                        for landmark in hand_landmarks:
                            fila_coordenadas.extend([landmark.x, landmark.y, landmark.z])
                        
                        datos_extraidos.append([letra] + fila_coordenadas)
                        muestras_tomadas += 1
                        # Pequeña pausa para evitar capturar frames exactamente idénticos
                        time.sleep(0.05) 

            cv2.imshow('Creando Dataset LSM', imagen)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(letra.lower()):
                grabando = True
                
            if key == 27: # ESC para salir
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

# 3. Guardar el archivo
if len(datos_extraidos) > 0:
    columnas = ['etiqueta']
    for i in range(21):
        columnas.extend([f'x{i}', f'y{i}', f'z{i}'])
    
    df = pd.DataFrame(datos_extraidos, columns=columnas)
    df.to_csv('dataset_lsm_coordenadas.csv', index=False)
    print("\nEXITO - Dataset real creado y guardado en 'dataset_lsm_coordenadas.csv'")
else:
    print("\nERROR - No se grabó ningún dato.")