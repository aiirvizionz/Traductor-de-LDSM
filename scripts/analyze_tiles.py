from pathlib import Path
from PIL import Image
import numpy as np
import pickle
base = Path(__file__).parents[1]
p = base / 'DATASET' / 'ABECEDARIOIMAGENES.pickle'
img = pickle.load(open(p,'rb'))
arr=np.array(img)
H,W,*_ = arr.shape
print('size',W,H)
# compute per-column variance across channels
col_var = arr.var(axis=(0,2)) if arr.ndim==3 else arr.var(axis=0)
row_var = arr.var(axis=(1,2)) if arr.ndim==3 else arr.var(axis=1)
# normalize
cv = (col_var - col_var.min())/(col_var.max()-col_var.min()+1e-9)
rv = (row_var - row_var.min())/(row_var.max()-row_var.min()+1e-9)
# find low-variance columns (separators)
col_seps = np.where(cv<0.01)[0]
row_seps = np.where(rv<0.01)[0]
print('num low-var cols:', len(col_seps))
print('num low-var rows:', len(row_seps))
# group contiguous indices
from itertools import groupby

def groups(idxs):
    groups=[]
    for k, g in groupby(enumerate(idxs), lambda x:x[0]-x[1]):
        grp=list(map(lambda x:x[1], g))
        groups.append((grp[0], grp[-1]))
    return groups

print('col sep groups:', groups(col_seps))
print('row sep groups:', groups(row_seps))

# find peaks in inverse variance to find separators roughly
try:
    import scipy.signal as sps
    peaks_c,_=sps.find_peaks(-cv, distance=5, prominence=0.5)
    peaks_r,_=sps.find_peaks(-rv, distance=5, prominence=0.5)
    print('peaks cols count:', len(peaks_c))
    print('peaks rows count:', len(peaks_r))
except Exception as e:
    print('scipy not available, skipping peak detection', e)

# Heurística: intentar divisores por factores de N (longitud de etiquetas si disponible)
try:
    etiquetas = pickle.load(open(base / 'DATASET' / 'ABECEDARIO.pickle','rb'))
    N = len(etiquetas)
except Exception:
    N = 72
factors=[(i,N//i) for i in range(1,N+1) if N % i==0]
print('possible grids:', factors)
for cols, rows in factors:
    if W % cols==0 and H % rows==0:
        print('grid fits exactly:', cols, 'x', rows, 'tile size', W//cols, H//rows)
    else:
        # fractional
        print('grid candidate:', cols, 'x', rows, 'tile size approx', W/cols, H/rows)

# print small sample pixels
print('top-left pixel:', arr[0,0])
print('center pixel:', arr[H//2, W//2])
