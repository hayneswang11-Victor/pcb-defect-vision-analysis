import pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, load_config

def main():
    cfg=load_config()
    src=ROOT/"outputs"/"yolo"; out=ROOT/"outputs"/"quality_analysis"
    out.mkdir(parents=True,exist_ok=True)
    cls=pd.read_csv(src/"class_metrics_quality_threshold.csv")
    scan=pd.read_csv(src/"threshold_scan.csv")
    errs=pd.read_csv(src/"error_cases_quality_threshold.csv")

    cls["error_total"]=cls["fp"]+cls["fn"]
    pareto=cls.sort_values("error_total",ascending=False).copy()
    total=pareto["error_total"].sum()
    pareto["error_share"]=pareto["error_total"]/total if total else 0
    pareto["cum_error_share"]=pareto["error_share"].cumsum()
    pareto.to_csv(out/"defect_error_pareto.csv",index=False,encoding="utf-8-sig")

    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(cls["class_name"],cls["recall"])
    ax.set_ylim(0,1.05); ax.set_ylabel("Recall"); ax.set_title("Recall by defect class")
    ax.tick_params(axis="x",rotation=25)
    fig.tight_layout(); fig.savefig(out/"class_recall.png",dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for col,label in [("precision","Precision"),("recall","Recall"),("f1","F1")]:
        ax.plot(scan["threshold"],scan[col],label=label)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Metric"); ax.set_ylim(0,1.05)
    ax.legend(); ax.set_title("Threshold trade-off")
    fig.tight_layout(); fig.savefig(out/"threshold_tradeoff.png",dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(pareto["class_name"],pareto["error_total"])
    ax.set_ylabel("FP + FN"); ax.set_title("Error Pareto by defect class")
    ax.tick_params(axis="x",rotation=25)
    fig.tight_layout(); fig.savefig(out/"error_pareto.png",dpi=160); plt.close(fig)

    top_n=int(cfg["evaluation"]["top_error_cases"])
    errs.head(top_n).to_csv(out/f"top_{top_n}_error_images.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{
        "macro_precision":cls["precision"].mean(),
        "macro_recall":cls["recall"].mean(),
        "macro_f1":cls["f1"].mean(),
        "total_fp":int(cls["fp"].sum()),
        "total_fn":int(cls["fn"].sum())
    }]).to_csv(out/"quality_summary.csv",index=False,encoding="utf-8-sig")
    print("质量分析输出完成：",out)

if __name__=="__main__":
    main()
