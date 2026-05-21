import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
print("Leyendo el dataset dinámico...")
ruta_datos = 'DATASET_DINAMICO'

acciones = np.array([
    'Adios', 'Bien', 'De nada', 'G', 'gracias', 'hola', 
    'J', 'No', 'Ñ', 'Perdon', 'Por favor', 
    'Q', 'Si', 'X', 'Z'
])
secuencias_por_accion = 30
frames_por_secuencia = 60
etiquetas_map = {label:num for num, label in enumerate(acciones)}

secuencias, etiquetas = [], []

for accion in acciones:
    for secuencia in range(secuencias_por_accion):
        ventana = []
        # Verificamos que la secuencia exista realmente
        if not os.path.exists(os.path.join(ruta_datos, accion, f"{secuencia}_0.npy")):
            continue
            
        for frame_num in range(frames_por_secuencia):
            resultado = np.load(os.path.join(ruta_datos, accion, f"{secuencia}_{frame_num}.npy"))
            ventana.append(resultado)
        
        secuencias.append(ventana)
        etiquetas.append(etiquetas_map[accion])

X = np.array(secuencias)
y = to_categorical(etiquetas).astype(int)

print(f"Dimensiones de X (Videos, Frames, Coordenadas): {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# 2. ARQUITECTURA LSTM LIGERA (Evita el Overfitting)
# ==========================================
print("\nConstruyendo red recurrente ligera...")

modelo = Sequential([
    # Una sola capa LSTM pequeña es suficiente para este volumen de datos
    LSTM(32, return_sequences=False, activation='relu', input_shape=(frames_por_secuencia, 63)),
    Dense(32, activation='relu'),
    # Aumentamos el Dropout a 30% para forzar al modelo a generalizar mejor
    Dropout(0.3),
    Dense(acciones.shape[0], activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==========================================
# 3. ENTRENAMIENTO INTELIGENTE
# ==========================================
print("\nIniciando entrenamiento temporal...")

# El callback vigila la precisión de prueba. Si pasan 15 iteraciones sin mejorar, se detiene solo.
parada_temprana = EarlyStopping(monitor='val_categorical_accuracy', patience=15, restore_best_weights=True)

historial = modelo.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    callbacks=[parada_temprana]
)

# ==========================================
# 4. EVALUACIÓN Y GUARDADO
# ==========================================
perdida, precision = modelo.evaluate(X_test, y_test)
print(f"\n Precision final validada del modelo LSTM: {precision * 100:.2f}%")

modelo.save('modelo_dinamico_lsm.keras')
np.save('clases_dinamicas.npy', acciones)
print("¡Modelo guardado!")