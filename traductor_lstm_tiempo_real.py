import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from collections import deque
from pathlib import Path

# ==========================================
# 1. CARGAR MODELO DINÁMICO
# ==========================================
print("Cargando el cerebro temporal LSTM...")
try:
    modelo = load_model('modelo_dinamico_lsm.keras')
    acciones = np.load('clases_dinamicas.npy', allow_pickle=True)
    print("¡Modelo LSTM cargado exitosamente!")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# ==========================================
# 2. CONFIGURAR MEDIAPIPE Y MEMORIA TEMPORAL
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

# 'memoria_video' guardará exactamente 60 frames. Si entra el 61, el 1 se borra automáticamente.
memoria_video = deque(maxlen=60)
palabra_actual = "..."
probabilidad = 0.0

# ==========================================
# 3. TRADUCCIÓN EN VIVO
# ==========================================
cap = cv2.VideoCapture(0)
print("\nIniciando traducción en vivo. Mueve tu mano naturalmente. Presiona 'ESC' para salir.")

with vision.HandLandmarker.create_from_options(opciones) as landmarker:
    while cap.isOpened():
        exito, imagen = cap.read()
        if not exito: continue
        
        H, W, _ = imagen.shape 
        imagen = cv2.flip(imagen, 1)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        imagen_mp_lista = np.ascontiguousarray(imagen_rgb)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_mp_lista)
        
        resultados = landmarker.detect(imagen_mp)

        fila_coordenadas = np.zeros(63)

        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                dibujar_mano(imagen, hand_landmarks)
                
                temporal = []
                for landmark in hand_landmarks:
                    temporal.extend([landmark.x, landmark.y, landmark.z])
                fila_coordenadas = np.array(temporal)
                
                # Normalización en tiempo real idéntica al entrenamiento
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

        # Agregamos el cuadro actual a la memoria (sea de ceros o con mano detectada)
        memoria_video.append(fila_coordenadas)

        # Solo predecimos si ya recolectamos 2 segundos completos de video
        if len(memoria_video) == 60:
            # Preparamos la matriz para la red neuronal: 1 video, 60 cuadros, 63 coordenadas
            entrada = np.array(memoria_video).reshape(1, 60, 63)
            
            # verbose=0 silencia la consola para que no se sature de texto
            predicciones = modelo.predict(entrada, verbose=0)[0]
            
            indice_ganador = np.argmax(predicciones)
            probabilidad = predicciones[indice_ganador]
            
            # Filtro de confianza: Solo mostrar la palabra si la IA está >75% segura
            if probabilidad > 0.75:
                palabra_actual = acciones[indice_ganador]

        # Interfaz Gráfica
        cv2.rectangle(imagen, (0, 0), (W, 85), (32, 32, 32), -1)
        cv2.putText(imagen, "Traductor Dinamico LSM (LSTM)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Barra de progreso visual para saber cuándo el buffer está lleno
        if len(memoria_video) < 60:
            cv2.putText(imagen, f"Llenando buffer temporal... {len(memoria_video)}/60", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            color_texto = (0, 255, 0) if probabilidad > 0.75 else (0, 0, 255)
            cv2.putText(imagen, f"{palabra_actual}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(imagen, f"{probabilidad*100:.0f}%", (W - 120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_texto, 2)

        cv2.imshow('Traductor LSTM', imagen)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()