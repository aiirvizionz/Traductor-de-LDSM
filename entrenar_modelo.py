import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
print("Cargando el dataset normalizado...")
df = pd.read_csv('dataset_lsm_normalizado.csv')

# Separar las características (X) de las etiquetas (Y)
X = df.drop('etiqueta', axis=1).values
Y_texto = df['etiqueta'].values

# Las redes neuronales no entienden letras ('A', 'B'), solo números.
# LabelEncoder convierte 'A' en 0, 'B' en 1, etc.
encoder = LabelEncoder()
Y_numerico = encoder.fit_transform(Y_texto)
num_clases = len(encoder.classes_)

print(f"Total de muestras: {len(X)}")
print(f"Clases a clasificar ({num_clases}): {encoder.classes_}")

# Dividir los datos: 80% para que la red estudie (entrenamiento) y 20% para el examen final (validación)
X_train, X_test, y_train, y_test = train_test_split(X, Y_numerico, test_size=0.2, random_state=42)

# ==========================================
# 2. CONSTRUCCIÓN DE LA RED NEURONAL
# ==========================================
print("\nConstruyendo la arquitectura de la red...")

modelo = Sequential([
    # Capa de entrada (63 variables) y Primera capa oculta
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    # Dropout apaga neuronas al azar para evitar que el modelo memorice (Overfitting)
    Dropout(0.2), 
    
    # Segunda capa oculta
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    # Tercera capa oculta más pequeña
    Dense(32, activation='relu'),
    
    # Capa de salida: una neurona por clase, activación Softmax
    Dense(num_clases, activation='softmax')
])

# Configurar cómo va a aprender la red
modelo.compile(optimizer='adam', 
               loss='sparse_categorical_crossentropy', 
               metrics=['accuracy'])

# Mostrar un resumen de la estructura
modelo.summary()

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
print("\nIniciando el entrenamiento...")
# epochs = las veces que repasará el dataset completo
# batch_size = la cantidad de datos que procesa a la vez antes de actualizarse
historial = modelo.fit(X_train, y_train, 
                       epochs=50, 
                       batch_size=32, 
                       validation_data=(X_test, y_test))

# ==========================================
# 4. EVALUACIÓN Y GUARDADO
# ==========================================
# Examen final con el 20% de los datos que el modelo nunca ha visto
perdida, precision = modelo.evaluate(X_test, y_test)
print(f"\nPrecisión final del modelo: {precision * 100:.2f}%")

# Guardar el "cerebro" entrenado para usarlo después con la cámara web
modelo.save('modelo_lsm.keras')

# Guardar también el diccionario de clases (qué número corresponde a qué letra)
# usando numpy
np.save('clases_lsm.npy', encoder.classes_)

print("¡Éxito! El modelo ('modelo_lsm.keras') y las etiquetas han sido guardados.")