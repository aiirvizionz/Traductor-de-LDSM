import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from pathlib import Path
import os

try:
    from PIL import Image
except ImportError:
    Image = None

# ==========================================
# 1. CARGA Y DIVISIÓN DE IMÁGENES (TILING INTELIGENTE)
# ==========================================
ruta_base = Path(__file__).parent
ruta_etiquetas = ruta_base / 'DATASET' / 'ABECEDARIO.pickle'
ruta_imagenes = ruta_base / 'DATASET' / 'ABECEDARIOIMAGENES.pickle'
MODELO_LOCAL = ruta_base / 'DATASET' / 'hand_landmarker.task'

carpeta_debug = ruta_base / 'DEBUG_TILES'
os.makedirs(carpeta_debug, exist_ok=True)

print("Cargando el dataset original...")
with open(ruta_etiquetas, 'rb') as f:
    etiquetas = pickle.load(f)
with open(ruta_imagenes, 'rb') as f:
    imagenes = pickle.load(f)

def centrar_y_recortar_mano(imagen_rgb, padding=30):
    """Usa visión por computadora para encontrar el contenido real y recortar el fondo blanco."""
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    
    # Asumimos fondo blanco. Todo lo que sea más oscuro (< 240) es la mano.
    _, mascara = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar los contornos del objeto detectado
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return imagen_rgb # Si está en blanco total, no hacemos nada
        
    # Encontrar la caja (bounding box) que encierre TODO el contenido
    x_min, y_min = imagen_rgb.shape[1], imagen_rgb.shape[0]
    x_max, y_max = 0, 0
    
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x + w > x_max: x_max = x + w
        if y + h > y_max: y_max = y + h
        
    # Aplicar el recorte con un margen seguro para no cortar dedos
    H, W = imagen_rgb.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding)
    y_max = min(H, y_max + padding)
    
    return imagen_rgb[y_min:y_max, x_min:x_max]

if Image is not None and isinstance(imagenes, Image.Image):
    img = imagenes
    
    # --- CORRECCIÓN VITAL PARA TRANSPARENCIAS ---
    # Si la imagen original tiene fondo transparente, lo rellenamos de blanco sólido
    if img.mode == 'RGBA':
        fondo = Image.new("RGB", img.size, (255, 255, 255))
        fondo.paste(img, mask=img.split()[3]) 
        img = fondo
    elif img.mode != 'RGB':
        img = img.convert('RGB')
        
    W, H = img.size
    N = len(etiquetas)
    
    factor_pairs = [(c, N // c) for c in range(1, N + 1) if N % c == 0]
    preferred = (8, 9) if (8, 9) in factor_pairs else (9, 8) if (9, 8) in factor_pairs else factor_pairs[0]
    cols, rows = preferred

    tile_w, tile_h = W // cols, H // rows
    tiles = []
    
    for r in range(rows):
        for c in range(cols):
            # 1. Cortar matemáticamente justo en los bordes de la celda (sin padding que jale el blanco)
            left, upper = c * tile_w, r * tile_h
            right, lower = left + tile_w, upper + tile_h

            tile_pil = img.crop((left, upper, right, lower))
            tile_np = np.array(tile_pil, dtype=np.uint8)
            
            # 2. Búsqueda inteligente: Quitar el exceso de blanco y hacer zoom a la mano
            tile_recortado = centrar_y_recortar_mano(tile_np, padding=20)
            
            # 3. Estandarizar tamaño 
            tile_final = cv2.resize(tile_recortado, (256, 256), interpolation=cv2.INTER_LINEAR)
            tiles.append(tile_final)

    imagenes = tiles
    print(f"Imagen dividida y auto-recortada en rejilla {cols}x{rows} -> {len(imagenes)} tiles.")

# ==========================================
# 2. EXTRACCIÓN CON MEDIAPIPE Y MEJORA VISUAL
# ==========================================
if not MODELO_LOCAL.exists():
    raise FileNotFoundError(f"¡Falta el archivo del modelo! '{MODELO_LOCAL}'")

base_options = python.BaseOptions(model_asset_path=str(MODELO_LOCAL))
opciones = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
)

datos_extraidos = []
manos_no_detectadas = 0

print(f"\nIniciando extraccion con MediaPipe Tasks para {len(imagenes)} imagenes...")
print("-" * 40)

with vision.HandLandmarker.create_from_options(opciones) as landmarker:
    for idx, (etiqueta, imagen_np) in enumerate(zip(etiquetas, imagenes)):
        
        imagen_mejorada = cv2.convertScaleAbs(imagen_np, alpha=1.2, beta=15)
        
        # Guardamos la imagen para que veas el nuevo recorte inteligente
        ruta_debug = str(carpeta_debug / f"tile_{idx}_{etiqueta}.jpg")
        cv2.imwrite(ruta_debug, cv2.cvtColor(imagen_mejorada, cv2.COLOR_RGB2BGR))

        imagen_mp_lista = np.ascontiguousarray(imagen_mejorada)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_mp_lista)

        resultados = landmarker.detect(imagen_mp)

        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                fila_coordenadas = []
                for landmark in hand_landmarks:
                    fila_coordenadas.extend([landmark.x, landmark.y, landmark.z])
                
                fila_completa = [etiqueta] + fila_coordenadas
                datos_extraidos.append(fila_completa)
        else:
            manos_no_detectadas += 1

print("-" * 40)
print(f"Extraccion finalizada.")
print(f"EXITO - Manos detectadas: {len(datos_extraidos)}")
print(f"IGNORADAS - Sin mano detectada: {manos_no_detectadas}")

# ==========================================
# 3. GUARDAR EL DATASET
# ==========================================
if len(datos_extraidos) > 0:
    columnas = ['etiqueta']
    for i in range(21):
        columnas.extend([f'x{i}', f'y{i}', f'z{i}'])

    df = pd.DataFrame(datos_extraidos, columns=columnas)
    nombre_archivo_salida = 'dataset_lsm_coordenadas.csv'
    df.to_csv(nombre_archivo_salida, index=False)
    print(f"\nExito! Datos guardados en el archivo: {nombre_archivo_salida}")
    print("REVISA LA CARPETA 'DEBUG_TILES' PARA VER LOS NUEVOS RECORTES.")
else:
    print("\nERROR: MediaPipe sigue sin detectar manos.")