from pathlib import Path
import pickle
from PIL import Image
import numpy as np

base = Path(__file__).parents[1]
p = base / 'DATASET' / 'ABECEDARIOIMAGENES.pickle'
img = pickle.load(open(p,'rb'))
assert isinstance(img, Image.Image)
W,H = img.size

# Determinar N (número de tiles) a partir de ABECEDARIO.pickle
try:
    etiquetas = pickle.load(open(base / 'DATASET' / 'ABECEDARIO.pickle','rb'))
    N = len(etiquetas)
except Exception:
    N = 72

# Preferir 8x9 o 12x6
factor_pairs = [(c, N // c) for c in range(1, N + 1) if N % c == 0]
if (8, 9) in factor_pairs:
    cols, rows = (8, 9)
elif (12, 6) in factor_pairs:
    cols, rows = (12, 6)
else:
    cols, rows = factor_pairs[0]

tile_w = W // cols
tile_h = H // rows
out = base / 'DATASET' / 'samples'
out.mkdir(parents=True, exist_ok=True)
count=0
for r in range(rows):
    for c in range(cols):
        left = c * tile_w
        upper = r * tile_h
        right = left + tile_w
        lower = upper + tile_h
        tile = img.crop((left, upper, right, lower))
        tile_path = out / f'tile_{r}_{c}.png'
        tile.save(tile_path)
        count+=1
        if count>=9:
            break
    if count>=9:
        break
print(f'Saved {count} sample tiles to {out}')
