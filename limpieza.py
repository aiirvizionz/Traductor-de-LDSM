import pickle
from pathlib import Path

# Cambia el nombre según el archivo que hayas descargado
BASE = Path(__file__).parent
ruta_archivo = BASE / 'DATASET' / 'ABECEDARIOIMAGENES.pickle'

try:
    with open(ruta_archivo, 'rb') as f:
        datos = pickle.load(f)
        
    print(f"Tipo de dato principal: {type(datos)}")
    
    # Si es un diccionario, veamos qué llaves (etiquetas) tiene
    if isinstance(datos, dict):
        print(f"Clases encontradas: {list(datos.keys())}")
        # Explorar el primer elemento de la primera llave
        primera_llave = list(datos.keys())[0]
        print(f"Muestra de la clase '{primera_llave}': {type(datos[primera_llave])}")
        
    # Si es una lista, veamos su longitud y el primer elemento
    elif isinstance(datos, list):
        print(f"Número total de muestras: {len(datos)}")
        print(f"Formato del primer elemento: {type(datos[0])}")

except Exception as e:
    print(f"Error al leer el archivo: {e}")