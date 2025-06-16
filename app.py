"""
ANN NumPy :
 - Recorta franja superior con texto
 - División estratificada
 - Pérdida ponderada por clase
 - Métricas 
"""
from __future__ import annotations
import itertools, random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# ----------------- Hiperparámetros -----------------
DATA_DIR       = Path("train")
TEST_DIR       = Path("test")
TARGET_SIZE    = (64, 64)
CROP_TOP_RATIO = 0.20        # ⚡ recortar 20 % superior
TRAIN_SPLIT    = 0.8
HIDDEN_UNITS   = 256         # ⚡ un poco más de capacidad
EPOCHS         = 60
BATCH_SIZE     = 32
LR             = 5e-3
SEED           = 42
# ---------------------------------------------------

np.random.seed(SEED); random.seed(SEED)

# ---------- Activaciones / utilidades -------------
def relu(x):            return np.maximum(0.0, x)
def drelu(x):           return (x > 0).astype(np.float32)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
def one_hot(y, C):      out=np.zeros((y.size, C), np.float32); out[np.arange(y.size), y]=1; return out

# ---------- Carga de datos con recorte ⚡ ----------
def preprocess(img: Image.Image) -> np.ndarray:
    w, h = img.size
    top_crop = int(CROP_TOP_RATIO * h)
    img = img.crop((0, top_crop, w, h))          # ⚡ recorte anti-texto
    img = img.resize(TARGET_SIZE)
    img = img.convert("L")                       # ⚡ escala grises
    return np.asarray(img, np.float32).flatten() / 255.0

def load_dataset(base: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, names = [], [], sorted(d.name for d in base.iterdir() if d.is_dir())
    for idx, cls in enumerate(names):
        for file in (base/cls).glob("*"):
            try:
                X.append(preprocess(Image.open(file)))
                y.append(idx)
            except Exception as e:
                print(f"[WARN] {file} omitida: {e}")
    return np.stack(X), np.array(y, int), names

# ---------- Stratified split ⚡ ---------------------
def stratified_split(X, y, ratio=TRAIN_SPLIT):
    idx_by_class = [np.where(y==c)[0] for c in range(len(np.unique(y)))]
    train_idx, val_idx = [], []
    for idxs in idx_by_class:
        np.random.shuffle(idxs)
        cut = int(len(idxs)*ratio)
        train_idx.extend(idxs[:cut]); val_idx.extend(idxs[cut:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# ---------- Pérdida ponderada ⚡ --------------------
def weighted_cross_entropy(pred, y_onehot, class_weights):
    eps=1e-9
    w = class_weights[y_onehot.argmax(1)]        # (N,)
    return -np.mean(w * np.sum(y_onehot * np.log(pred+eps),1))

# ---------- Modelo --------------------------------
class ANN:
    def __init__(self, D, H, C):
        self.W1 = np.random.randn(D,H).astype(np.float32)*np.sqrt(2/D)
        self.b1 = np.zeros((1,H),np.float32)
        self.W2 = np.random.randn(H,C).astype(np.float32)*np.sqrt(2/H)
        self.b2 = np.zeros((1,C),np.float32)
    def forward(self,X):
        z1 = X@self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1@self.W2 + self.b2
        a2 = softmax(z2)
        return z1,a1,a2
    def backward(self,X,y1,z1,a1,a2,lr,class_w):
        m=X.shape[0]
        dz2 = (a2-y1)/m * class_w[y1.argmax(1),None]  # ⚡ ponderación
        self.W2 -= lr * (a1.T@dz2)
        self.b2 -= lr * dz2.sum(0,keepdims=True)
        dz1 = (dz2@self.W2.T)*drelu(z1)
        self.W1 -= lr*(X.T@dz1)
        self.b1 -= lr*dz1.sum(0,keepdims=True)
    def predict(self,X): return self.forward(X)[-1].argmax(1)

# ---------- Matriz de confusión ⚡ ------------------
def confusion_matrix(y_true,y_pred,C):
    mat=np.zeros((C,C),int)
    for t,p in zip(y_true,y_pred): mat[t,p]+=1
    return mat
def print_conf(mat, names):
    print("Confusion matrix:")
    rows=[]
    for i,r in enumerate(mat):
        rows.append(" ".join(f"{n:>3}" for n in r)+f" | {names[i]}")
    print("\n".join(rows))

# ---------- Entrenamiento principal ----------------
def train():
    X,y,names = load_dataset(DATA_DIR)
    Xtr,ytr,Xv,yv = stratified_split(X,y)
    print("Clases:",names,"| balance:",np.bincount(y))

    C=len(names); D=X.shape[1]; H=HIDDEN_UNITS
    class_weights = np.bincount(ytr, minlength=C).astype(np.float32)
    class_weights = class_weights.mean()/class_weights   # inverso de frecuencia ⚡
    model=ANN(D,H,C)

    ytr_oh,yv_oh = one_hot(ytr,C), one_hot(yv,C)
    for ep in range(1,EPOCHS+1):
        idx=np.random.permutation(len(Xtr))
        for b in range(0,len(idx),BATCH_SIZE):
            j=idx[b:b+BATCH_SIZE]
            z1,a1,a2=model.forward(Xtr[j])
            model.backward(Xtr[j], ytr_oh[j], z1,a1,a2, LR, class_weights)
        # métricas
        pred_tr=model.predict(Xtr); pred_v=model.predict(Xv)
        acc_tr=(pred_tr==ytr).mean(); acc_v=(pred_v==yv).mean()
        if ep%5==0 or ep==1:
            print(f"Ep{ep:03d} acc {acc_tr:.2%}/{acc_v:.2%}")
            print_conf(confusion_matrix(yv,pred_v,C), names)

    # ---------- inferencia en /test ----------
    print("\n=== /test predictions ===")
    for p in sorted(TEST_DIR.glob("*")):
        try:
            x=preprocess(Image.open(p))[None,:]
            print(f"{p.name:25} → {names[model.predict(x)[0]]}")
        except Exception as e:
            print("[WARN]",p,e)

if __name__=="__main__":
    train()
