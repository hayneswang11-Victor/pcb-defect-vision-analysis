from pathlib import Path
import random, yaml, numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT/"configs"/"project.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT/p

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def box_iou_xyxy(a,b):
    ax1,ay1,ax2,ay2 = map(float,a)
    bx1,by1,bx2,by2 = map(float,b)
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1)
    inter=iw*ih
    aa=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    bb=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    union=aa+bb-inter
    return inter/union if union>0 else 0.0
