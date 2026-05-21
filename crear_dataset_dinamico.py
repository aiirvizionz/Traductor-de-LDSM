import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN DEL DATASET DINÁMICO
# ==========================================
acciones = np.array([
    'G', 'J', 'N_asento', 'Q', 'X', 'Z', 
    'Hola', 'Gracias', 'Por favor', 'Si', 'No', 'Adios', 'De nada', 'Bien', 'Perdon'
]) 

secuencias_por_accion = 30  
# 60 frames equivalen aproximadamente a 2 segundos de video en una cámara estándar
frames_por_secuencia = 60   

ruta_datos = os.path.join('DATASET_DINAMICO')

for accion in acciones:
    try:
        os.makedirs(os.path.join(ruta_datos, accion))
    except OSError:
        pass

# ==========================================
# 2. CONFIGURACIÓN DE MEDIAPIPE
# ==========================================
ruta_base = Path(__file__).parent
MODELO_LOCAL = ruta_base / 'DATASET' / 'hand_landmarker.task'

opciones = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODELO_LOCAL)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

def dibujar_mano(imagen, hand_landmarks):
    H, W, _ = imagen.shape
    conexiones = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
                  (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
                  (13,17),(17,18),(18,19),(19,20),(0,17)]
    for inicio, fin in conexiones:
        x1, y1 = int(hand_landmarks[inicio].x * W), int(hand_landmarks[inicio].y * H)
        x2, y2 = int(hand_landmarks[fin].x * W), int(hand_landmarks[fin].y * H)
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in hand_landmarks:
        x, y = int(lm.x * W), int(lm.y * H)
        cv2.circle(imagen, (x, y), 5, (0, 0, 255), -1)

# ==========================================
# 3. CAPTURA DE SECUENCIAS
# ==========================================
cap = cv2.VideoCapture(0)
print("Iniciando cámara para dataset temporal...")

with vision.HandLandmarker.create_from_options(opciones) as landmarker:
    for accion in acciones:
        for secuencia in range(secuencias_por_accion):
            
            # --- NUEVO: BUCLE DE ESPERA MANUAL ---
            # La cámara se queda en vivo mostrándote, esperando la barra espaciadora
            while True:
                exito, imagen = cap.read()
                if not exito: continue
                
                imagen = cv2.flip(imagen, 1)
                cv2.putText(imagen, f'SENA: {accion}', (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(imagen, f'Presiona ESPACIO para iniciar Video {secuencia + 1}/{secuencias_por_accion}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('Recoleccion LSM Dinamico', imagen)
                
                key = cv2.waitKey(10) & 0xFF
                if key == 32: # 32 es el código de la barra espaciadora
                    break
                if key == 27: # 27 es la tecla ESC para salir
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                    
            # --- INICIA GRABACIÓN DE 2 SEGUNDOS ---
            for frame_num in range(frames_por_secuencia):
                exito, imagen = cap.read()
                if not exito: continue
                
                imagen = cv2.flip(imagen, 1)
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                
                imagen_mp_lista = np.ascontiguousarray(imagen_rgb)
                imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_mp_lista)
                
                resultados = landmarker.detect(imagen_mp)

                # Interfaz de grabación
                cv2.putText(imagen, f'GRABANDO {accion} - Video {secuencia + 1}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                fila_coordenadas = np.zeros(63) 
                
                if resultados.hand_landmarks:
                    for hand_landmarks in resultados.hand_landmarks:
                        dibujar_mano(imagen, hand_landmarks)
                        temporal = []
                        for landmark in hand_landmarks:
                            temporal.extend([landmark.x, landmark.y, landmark.z])
                        fila_coordenadas = np.array(temporal)
                
                if np.sum(fila_coordenadas) != 0:
                    puntos = fila_coordenadas.reshape(-1, 3)
                    origen = puntos[0]
                    puntos_centrados = puntos - origen
                    max_valor = np.max(np.abs(puntos_centrados))
                    if max_valor > 0:
                        puntos_escalados = puntos_centrados / max_valor
                    else:
                        puntos_escalados = puntos_centrados
                    fila_coordenadas = puntos_escalados.flatten()

                ruta_frame = os.path.join(ruta_datos, accion, f"{secuencia}_{frame_num}.npy")
                np.save(ruta_frame, fila_coordenadas)
                
                cv2.imshow('Recoleccion LSM Dinamico', imagen)
                
                if cv2.waitKey(10) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
print("\n¡Dataset Dinámico recolectado con éxito!")