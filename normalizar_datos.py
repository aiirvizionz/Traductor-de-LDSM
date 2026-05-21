import pandas as pd
import numpy as np

# 1. Cargar el dataset con las coordenadas crudas
archivo_entrada = 'dataset_lsm_coordenadas.csv'
print(f"Cargando datos crudos desde: {archivo_entrada}")
df = pd.read_csv(archivo_entrada)

# Separar la columna de etiquetas (las letras) de los números
etiquetas = df['etiqueta']
# Convertimos solo los números a una matriz de NumPy para hacer la matemática más rápido
coordenadas = df.drop('etiqueta', axis=1).values 

datos_normalizados = []

print("Iniciando proceso de normalización (Traslación y Escalado)...")

# 2. Iterar sobre cada fila (cada imagen de una mano)
for fila in coordenadas:
    # Reestructuramos la fila plana de 63 números a una matriz de (21 puntos, 3 ejes: x,y,z)
    puntos = fila.reshape(-1, 3)
    
    # --- PASO A: TRASLACIÓN (Centrar en la muñeca) ---
    # El punto 0 de MediaPipe siempre es la base de la muñeca
    origen_muneca = puntos[0]
    
    # Al restar, la muñeca se vuelve (0,0,0) y los demás puntos se miden en relación a ella
    puntos_centrados = puntos - origen_muneca
    
    # --- PASO B: ESCALADO (Estandarizar tamaño) ---
    # Buscamos el valor de distancia más grande en esta mano en específico
    max_valor = np.max(np.abs(puntos_centrados))
    
    # Dividimos todos los puntos entre el valor máximo para que queden entre -1.0 y 1.0
    # Agregamos una pequeña validación para no dividir entre cero por accidente
    if max_valor > 0:
        puntos_escalados = puntos_centrados / max_valor
    else:
        puntos_escalados = puntos_centrados
        
    # Volvemos a aplanar la matriz de (21, 3) a una sola fila de 63 números
    datos_normalizados.append(puntos_escalados.flatten())

# 3. Ensamblar y guardar el nuevo dataset limpio
print("Ensamblando el nuevo archivo...")

# Recuperamos los nombres de las columnas originales (x0, y0, z0...)
nombres_columnas = df.columns[1:] 

# Creamos un nuevo DataFrame con los números ya normalizados
df_normalizado = pd.DataFrame(datos_normalizados, columns=nombres_columnas)

# Le volvemos a pegar la columna de 'etiqueta' al principio
df_normalizado.insert(0, 'etiqueta', etiquetas)

# Guardamos el resultado final
archivo_salida = 'dataset_lsm_normalizado.csv'
df_normalizado.to_csv(archivo_salida, index=False)

print(f"¡Normalización exitosa! Archivo listo para la red neuronal: {archivo_salida}")