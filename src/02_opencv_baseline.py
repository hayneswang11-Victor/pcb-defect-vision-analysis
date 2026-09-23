from pathlib import Path
import cv2
import pandas as pd
from common import ROOT, load_config, resolve, box_iou_xyxy

def parse_annotation_line(line):
    """
    DeepPCB 官方标注实际采用空白分隔：
        x1 y1 x2 y2 class_id
    同时兼容逗号分隔版本。
    """
    line = line.strip()
    if not line:
        return None

    parts = line.replace(",", " ").split()
    if len(parts) != 5:
        raise ValueError(f"bad_annotation_field_count: {line}")

    x1, y1, x2, y2, c = map(int, parts)
    return x1, y1, x2, y2, c

def load_gt(path):
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue

        parsed = parse_annotation_line(line)
        if parsed is None:
            continue

        x1, y1, x2, y2, c = parsed
        out.append((x1, y1, x2, y2, c))

    return out

def detect(test, temp, cfg):
    g1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    k = int(cfg["opencv"]["blur_kernel"])
    if k > 1:
        if k % 2 == 0:
            k += 1
        g1 = cv2.GaussianBlur(g1, (k, k), 0)
        g2 = cv2.GaussianBlur(g2, (k, k), 0)

    diff = cv2.absdiff(g1, g2)

    if cfg["opencv"]["use_otsu"]:
        _, mask = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, mask = cv2.threshold(
            diff,
            int(cfg["opencv"]["fixed_threshold"]),
            255,
            cv2.THRESH_BINARY,
        )

    mk = int(cfg["opencv"]["morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        area = cv2.contourArea(c)

        if (
            float(cfg["opencv"]["min_area"])
            <= area
            <= float(cfg["opencv"]["max_area"])
        ):
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))

    return boxes, mask, diff

def match(preds, gts, iou_thr):
    used = set()
    tp = 0

    for p in preds:
        best_iou = 0.0
        best = None

        for i, g in enumerate(gts):
            if i in used:
                continue

            iou = box_iou_xyxy(p, g[:4])

            if iou > best_iou:
                best_iou = iou
                best = i

        if best is not None and best_iou >= iou_thr:
            used.add(best)
            tp += 1

    return tp, len(preds) - tp, len(gts) - tp

def main():
    cfg = load_config()
    prepared = resolve(cfg["dataset"]["prepared_root"])

    out = ROOT / "outputs" / "opencv_baseline"
    vis = out / "visualizations"

    out.mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)

    rows = []
    TP = FP = FN = 0
    thr = float(cfg["opencv"]["iou_threshold"])

    test_paths = sorted((prepared / "images" / "test").glob("*.jpg"))

    if not test_paths:
        raise RuntimeError(
            "没有找到 prepared test images。请先运行 01_prepare_yolo_dataset.py。"
        )

    for idx, test_path in enumerate(test_paths):
        sid = test_path.stem

        temp_path = prepared / "templates" / "test" / f"{sid}.jpg"
        gt_path = prepared / "gt_original" / "test" / f"{sid}.txt"

        test = cv2.imread(str(test_path))
        temp = cv2.imread(str(temp_path))

        if test is None or temp is None:
            rows.append({
                "sample_id": sid,
                "status": "read_error",
                "gt_boxes": None,
                "pred_boxes": None,
                "tp": None,
                "fp": None,
                "fn": None,
            })
            continue

        if not gt_path.exists():
            rows.append({
                "sample_id": sid,
                "status": "missing_gt",
                "gt_boxes": None,
                "pred_boxes": None,
                "tp": None,
                "fp": None,
                "fn": None,
            })
            continue

        preds, mask, diff = detect(test, temp, cfg)
        gts = load_gt(gt_path)

        tp, fp, fn = match(preds, gts, thr)

        TP += tp
        FP += fp
        FN += fn

        rows.append({
            "sample_id": sid,
            "status": "ok",
            "gt_boxes": len(gts),
            "pred_boxes": len(preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        if idx < 20:
            canvas = test.copy()

            for x1, y1, x2, y2, *_ in gts:
                cv2.rectangle(
                    canvas, (x1, y1), (x2, y2), (0, 255, 0), 1
                )

            for x1, y1, x2, y2 in preds:
                cv2.rectangle(
                    canvas, (x1, y1), (x2, y2), (0, 0, 255), 1
                )

            cv2.imwrite(str(vis / f"{sid}_boxes.jpg"), canvas)
            cv2.imwrite(str(vis / f"{sid}_mask.png"), mask)
            cv2.imwrite(str(vis / f"{sid}_diff.png"), diff)

    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    pd.DataFrame(rows).to_csv(
        out / "image_level_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([{
        "iou_threshold": thr,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "precision": p,
        "recall": r,
        "f1": f1,
        "note": "类别无关的缺陷候选定位基线",
    }]).to_csv(
        out / "baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ok_count = sum(1 for row in rows if row["status"] == "ok")

    print(f"处理测试图：{len(test_paths)}")
    print(f"成功评估：{ok_count}")
    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"OpenCV baseline: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    main()
