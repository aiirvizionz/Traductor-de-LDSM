from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODELO_LOCAL = Path(__file__).parent / "DATASET" / "hand_landmarker.task"


def descargar_modelo_si_hace_falta() -> Path:
    MODELO_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if not MODELO_LOCAL.exists():
        print("Descargando modelo de detección de manos...")
        urllib.request.urlretrieve(MODELO_URL, MODELO_LOCAL)
    return MODELO_LOCAL


def main() -> None:
    modelo = descargar_modelo_si_hace_falta()

    opciones = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(modelo)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    print("Presiona 'Esc' en la ventana de video para salir.")

    try:
        with vision.HandLandmarker.create_from_options(opciones) as landmarker:
            while cap.isOpened():
                exito, imagen = cap.read()
                if not exito:
                    print("Ignorando frame de cámara vacío.")
                    continue

                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                imagen_mp = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=imagen_rgb,
                )
                timestamp_ms = int(time.time() * 1000)
                resultados = landmarker.detect_for_video(imagen_mp, timestamp_ms)

                if resultados.hand_landmarks:
                    for hand_landmarks in resultados.hand_landmarks:
                        vision.drawing_utils.draw_landmarks(
                            imagen,
                            hand_landmarks,
                            vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        )

                cv2.imshow("Traductor LSM - Extracción de Características", imagen)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()