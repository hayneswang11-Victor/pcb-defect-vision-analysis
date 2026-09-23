import torch
from ultralytics import YOLO
from common import load_config, resolve, set_seed

def main():
    cfg=load_config(); y=cfg["yolo"]; set_seed(int(cfg["project"]["seed"]))
    data=resolve(cfg["dataset"]["prepared_root"])/"dataset.yaml"
    if not data.exists(): raise FileNotFoundError("请先运行 01_prepare_yolo_dataset.py。")
    device=y["device"]
    if device=="auto": device=0 if torch.cuda.is_available() else "cpu"
    model=YOLO(y["model"])
    model.train(
        data=str(data), epochs=int(y["epochs"]), imgsz=int(y["imgsz"]),
        batch=int(y["batch"]), workers=int(y["workers"]), device=device,
        patience=int(y["patience"]), project=str(resolve(y["project_dir"])),
        name=y["run_name"], exist_ok=True, seed=int(cfg["project"]["seed"]),
        deterministic=True, plots=True, verbose=True
    )
    print("最佳权重：",resolve(y["project_dir"])/y["run_name"]/"weights"/"best.pt")

if __name__=="__main__":
    main()
