import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from pathlib import Path

# ==========================================
# 1. CARGAR CEREBRO ARTIFICIAL Y ETIQUETAS
# ==========================================
print("Cargando modelo de red neuronal...")
try:
    modelo = load_model('modelo_lsm.keras')
    clases = np.load('clases_lsm.npy', allow_pickle=True)
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error al cargar el modelo. ¿Ya ejecutaste entrenar_modelo.py? Detalles: {e}")
    exit()

# ==========================================
# 2. CONFIGURAR MEDIAPIPE
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

# Función visual
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
# 3. TRADUCCIÓN EN VIVO
# ==========================================
cap = cv2.VideoCapture(0)
print("\nIniciando cámara para traducción. Presiona 'ESC' para salir.")

with vision.HandLandmarker.create_from_options(opciones) as landmarker:
    while cap.isOpened():
        exito, imagen = cap.read()
        if not exito:
            continue
            
        # --- LÍNEA NUEVA A AGREGAR ---
        # Extraemos el Alto (H) y el Ancho (W) de la cámara
        H, W, _ = imagen.shape 
        # -----------------------------

        imagen = cv2.flip(imagen, 1)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        imagen_mp_lista = np.ascontiguousarray(imagen_rgb)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_mp_lista)
        
        resultados = landmarker.detect(imagen_mp)

        # Letra por defecto si no hay mano
        letra_predicha = "..."
        probabilidad = 0.0

        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                dibujar_mano(imagen, hand_landmarks)
                
                # A. Extraer coordenadas crudas
                fila_coordenadas = []
                for landmark in hand_landmarks:
                    fila_coordenadas.extend([landmark.x, landmark.y, landmark.z])
                
                # B. Normalizar exactamente igual que en normalizar_datos.py
                puntos = np.array(fila_coordenadas).reshape(-1, 3)
                origen_muneca = puntos[0]
                puntos_centrados = puntos - origen_muneca
                max_valor = np.max(np.abs(puntos_centrados))
                
                if max_valor > 0:
                    puntos_escalados = puntos_centrados / max_valor
                else:
                    puntos_escalados = puntos_centrados
                
                # Aplanar el vector para la red neuronal (1 fila, 63 columnas)
                entrada_red = puntos_escalados.flatten().reshape(1, -1)
                
                # C. Predicción de TensorFlow
                predicciones = modelo.predict(entrada_red, verbose=0)
                indice_clase = np.argmax(predicciones[0])
                probabilidad = predicciones[0][indice_clase]
                
                # Mostrar solo si la IA está segura (más del 60% de probabilidad)
                if probabilidad > 0.60:
                    letra_predicha = clases[indice_clase]

        # Interfaz de usuario gráfica
        cv2.rectangle(imagen, (0, 0), (W, 80), (245, 117, 16), -1)
        cv2.putText(imagen, "TRADUCCION LSM:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(imagen, f"{letra_predicha}  ({probabilidad*100:.1f}%)", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow('Prototipo LSM a Texto', imagen)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()