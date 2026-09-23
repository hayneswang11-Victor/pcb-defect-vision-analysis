from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from ultralytics import YOLO
from common import ROOT, load_config, resolve, box_iou_xyxy

def load_gt(label_path, size):
    rows=[]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        c,xc,yc,bw,bh=map(float,line.split())
        x1=(xc-bw/2)*size; y1=(yc-bh/2)*size
        x2=(xc+bw/2)*size; y2=(yc+bh/2)*size
        rows.append((int(c),x1,y1,x2,y2))
    return rows

def match(preds,gts,conf_thr,iou_thr):
    kept=[p for p in preds if p["conf"]>=conf_thr]
    kept.sort(key=lambda x:x["conf"],reverse=True)
    used=set(); tp=[]; fp=[]
    for p in kept:
        best_iou=0; best=None
        for gi,g in enumerate(gts):
            if gi in used or g[0]!=p["class_id"]: continue
            iou=box_iou_xyxy(p["box"],g[1:5])
            if iou>best_iou: best_iou,best=iou,gi
        if best is not None and best_iou>=iou_thr:
            used.add(best); tp.append((p,best,best_iou))
        else:
            fp.append(p)
    fn=[g for gi,g in enumerate(gts) if gi not in used]
    return tp,fp,fn

def aggregate(pred_by,gt_by,conf_thr,iou_thr,n_classes=6):
    cls={i:{"tp":0,"fp":0,"fn":0} for i in range(n_classes)}
    image_errors=[]
    for sid,gts in gt_by.items():
        tp,fp,fn=match(pred_by.get(sid,[]),gts,conf_thr,iou_thr)
        for p,_,_ in tp: cls[p["class_id"]]["tp"]+=1
        for p in fp: cls[p["class_id"]]["fp"]+=1
        for g in fn: cls[g[0]]["fn"]+=1
        if fp or fn: image_errors.append((sid,len(fp),len(fn)))
    T=sum(v["tp"] for v in cls.values())
    FP=sum(v["fp"] for v in cls.values())
    FN=sum(v["fn"] for v in cls.values())
    p=T/(T+FP) if T+FP else 0
    r=T/(T+FN) if T+FN else 0
    f1=2*p*r/(p+r) if p+r else 0
    return p,r,f1,cls,image_errors

def main():
    cfg=load_config(); y=cfg["yolo"]; e=cfg["evaluation"]
    prepared=resolve(cfg["dataset"]["prepared_root"])
    out=ROOT/"outputs"/"yolo"; out.mkdir(parents=True,exist_ok=True)
    weights=resolve(y["project_dir"])/y["run_name"]/"weights"/"best.pt"
    if not weights.exists(): raise FileNotFoundError(f"未找到：{weights}")
    model=YOLO(str(weights))

    metrics=model.val(data=str(prepared/"dataset.yaml"),split="test",
                      imgsz=int(y["imgsz"]),conf=0.001,iou=0.60,
                      plots=True,project=str(out),name="validation",
                      exist_ok=True,verbose=False)
    pd.DataFrame([{
        "precision_mean":float(metrics.box.mp),
        "recall_mean":float(metrics.box.mr),
        "mAP50":float(metrics.box.map50),
        "mAP50_95":float(metrics.box.map)
    }]).to_csv(out/"yolo_standard_metrics.csv",index=False,encoding="utf-8-sig")

    pred_by=defaultdict(list); pred_rows=[]
    results=model.predict(source=str(prepared/"images"/"test"),
                          imgsz=int(y["imgsz"]),conf=float(e["threshold_start"]),
                          iou=0.60,save=False,verbose=False,stream=True)
    for res in results:
        sid=Path(res.path).stem
        if res.boxes is None: continue
        for xyxy,conf,c in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy().astype(int)):
            box=list(map(float,xyxy))
            row={"sample_id":sid,"class_id":int(c),"conf":float(conf),
                 "x1":box[0],"y1":box[1],"x2":box[2],"y2":box[3],"box":box}
            pred_by[sid].append(row)
            pred_rows.append({k:v for k,v in row.items() if k!="box"})
    pd.DataFrame(pred_rows).to_csv(out/"all_predictions_low_threshold.csv",index=False,encoding="utf-8-sig")

    size=int(y["imgsz"]); gt_by={}
    for p in sorted((prepared/"labels"/"test").glob("*.txt")):
        gt_by[p.stem]=load_gt(p,size)

    thresholds=np.round(np.arange(float(e["threshold_start"]),
                                  float(e["threshold_end"])+1e-9,
                                  float(e["threshold_step"])),4)
    scan=[]
    for th in thresholds:
        p,r,f1,_,_=aggregate(pred_by,gt_by,float(th),float(e["match_iou"]))
        scan.append({"threshold":th,"precision":p,"recall":r,"f1":f1})
    scan_df=pd.DataFrame(scan)
    scan_df.to_csv(out/"threshold_scan.csv",index=False,encoding="utf-8-sig")

    best=scan_df.sort_values(["f1","recall"],ascending=False).iloc[0]
    target=float(e["target_recall"])
    cand=scan_df[scan_df["recall"]>=target]
    quality=(cand.sort_values(["threshold","precision"],ascending=False).iloc[0]
             if len(cand) else
             scan_df.sort_values(["recall","precision"],ascending=False).iloc[0])
    pd.DataFrame([
        {"mode":"best_f1","threshold":best.threshold,"precision":best.precision,
         "recall":best.recall,"f1":best.f1},
        {"mode":"quality_high_recall","threshold":quality.threshold,
         "precision":quality.precision,"recall":quality.recall,"f1":quality.f1}
    ]).to_csv(out/"selected_thresholds.csv",index=False,encoding="utf-8-sig")

    _,_,_,stats,errors=aggregate(pred_by,gt_by,float(quality.threshold),float(e["match_iou"]))
    names=cfg["dataset"]["classes"]; rows=[]
    for c,s in stats.items():
        p=s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"] else 0
        r=s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"] else 0
        f1=2*p*r/(p+r) if p+r else 0
        rows.append({"class_id":c,"class_name":names[c],"tp":s["tp"],"fp":s["fp"],
                     "fn":s["fn"],"precision":p,"recall":r,"f1":f1})
    pd.DataFrame(rows).to_csv(out/"class_metrics_quality_threshold.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(errors,columns=["sample_id","fp","fn"]).sort_values(
        ["fn","fp"],ascending=False).to_csv(
        out/"error_cases_quality_threshold.csv",index=False,encoding="utf-8-sig")

    print("Best F1 工作点：",dict(best))
    print("高召回工作点：",dict(quality))

if __name__=="__main__":
    main()
