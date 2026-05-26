# Traductor de Lengua de Señas Mexicana (LSM) con Deep Learning 🤟

Este proyecto es un sistema de Visión por Computadora e Inteligencia Artificial capaz de traducir la Lengua de Señas Mexicana (LSM) a texto en tiempo real utilizando una cámara web. 

El sistema está dividido en dos arquitecturas principales para abordar tanto señas estáticas (fotografías) como señas dinámicas (secuencias de video).

## 🧠 Arquitectura del Proyecto

### 1. Modelo Estático (Abecedario)
Utiliza una **Red Neuronal Densa (DNN)** para clasificar señas que no requieren movimiento (ej. A, B, C). 
* Extrae 21 puntos clave de la mano usando `MediaPipe Tasks API`.
* Normaliza las coordenadas matemáticamente (traslación al origen y escalado).
* Traduce el abecedario básico en tiempo real.

### 2. Modelo Dinámico (Palabras y Letras en Movimiento)
Implementa una arquitectura **LSTM (Long Short-Term Memory)** para comprender la evolución temporal de las señas que requieren movimiento (ej. J, Z, Hola, Gracias).
* Recolecta tensores tridimensionales de 60 cuadros (2 segundos de video) por seña.
* Entrena una red recurrente que analiza la secuencia completa antes de emitir una predicción.
* Evita el sobreajuste mediante capas de Dropout y callbacks de Early Stopping.

## 🛠️ Tecnologías Utilizadas
* **Python 3.12**
* **TensorFlow / Keras:** Construcción y entrenamiento de los modelos de Deep Learning (Dense y LSTM).
* **MediaPipe (Tasks API):** Extracción robusta de *Hand Landmarks*.
* **OpenCV:** Procesamiento de imágenes y renderizado de la interfaz en tiempo real.
* **NumPy & Pandas:** Estructuración y normalización matemática de los tensores espaciales.
* **Scikit-Learn:** Partición de datos (Train/Test Split).

## 📂 Estructura de Archivos Principales
* `crear_dataset_lsm.py` / `crear_dataset_dinamico.py`: Scripts con interfaz gráfica para generar bases de datos personalizadas vía webcam.
* `entrenar_modelo.py` / `entrenar_lstm.py`: Pipelines de entrenamiento y compilación de los modelos `.keras`.
* `traductor_tiempo_real.py` / `traductor_lstm_tiempo_real.py`: Scripts de ejecución final que conectan la cámara web con la red neuronal para la traducción en vivo.

## 🚀 Cómo ejecutarlo

1. Clona este repositorio.
2. Instala las dependencias necesarias:
   ```bash
   pip install opencv-python mediapipe tensorflow pandas numpy scikit-learn
