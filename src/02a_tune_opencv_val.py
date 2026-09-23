from pathlib import Path
import itertools
import cv2
import pandas as pd
import yaml

from common import ROOT, load_config, resolve, box_iou_xyxy


def parse_annotation_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.replace(",", " ").split()
    if len(parts) != 5:
        raise ValueError("bad_annotation_field_count: %s" % line)
    return tuple(map(int, parts))


def load_gt(path):
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parsed = parse_annotation_line(line)
        if parsed is not None:
            boxes.append(parsed)
    return boxes


def detect_candidates(test_bgr, template_bgr, fixed_threshold,
                      min_area, morph_kernel, blur_kernel, max_area):
    g1 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    blur_kernel = int(blur_kernel)
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        g1 = cv2.GaussianBlur(g1, (blur_kernel, blur_kernel), 0)
        g2 = cv2.GaussianBlur(g2, (blur_kernel, blur_kernel), 0)

    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(
        diff, int(fixed_threshold), 255, cv2.THRESH_BINARY
    )

    morph_kernel = int(morph_kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel, morph_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if float(min_area) <= area <= float(max_area):
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
    return boxes


def greedy_match(pred_boxes, gt_boxes, iou_threshold):
    used_gt = set()
    tp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt:
                continue
            iou = box_iou_xyxy(pred, gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None and best_iou >= float(iou_threshold):
            used_gt.add(best_idx)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def evaluate(samples, fixed_threshold, min_area, morph_kernel,
             blur_kernel, max_area, iou_threshold):
    tp_total = fp_total = fn_total = 0

    for s in samples:
        preds = detect_candidates(
            s["test"], s["template"],
            fixed_threshold, min_area, morph_kernel,
            blur_kernel, max_area
        )
        tp, fp, fn = greedy_match(preds, s["gt"], iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / float(tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / float(tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "fixed_threshold": int(fixed_threshold),
        "min_area": int(min_area),
        "morph_kernel": int(morph_kernel),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    cfg = load_config()
    prepared = resolve(cfg["dataset"]["prepared_root"])

    image_dir = prepared / "images" / "val"
    template_dir = prepared / "templates" / "val"
    gt_dir = prepared / "gt_original" / "val"

    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError("val 集为空，请先运行 01_prepare_yolo_dataset.py。")

    blur_kernel = int(cfg["opencv"]["blur_kernel"])
    max_area = int(cfg["opencv"]["max_area"])
    iou_threshold = float(cfg["opencv"]["iou_threshold"])

    fixed_thresholds = [20, 30, 40, 50, 60]
    min_areas = [20, 50, 100]
    morph_kernels = [3, 5]

    samples = []
    print("加载 val 数据...")

    for test_path in paths:
        sid = test_path.stem
        template_path = template_dir / ("%s.jpg" % sid)
        gt_path = gt_dir / ("%s.txt" % sid)

        if not template_path.exists():
            raise FileNotFoundError("缺少模板图：%s" % template_path)
        if not gt_path.exists():
            raise FileNotFoundError("缺少标注：%s" % gt_path)

        test = cv2.imread(str(test_path))
        template = cv2.imread(str(template_path))
        if test is None or template is None:
            raise RuntimeError("图像读取失败：%s" % sid)

        samples.append({
            "sample_id": sid,
            "test": test,
            "template": template,
            "gt": load_gt(gt_path),
        })

    print("val 图像：%d" % len(samples))
    print("参数组合：30")

    records = []
    combos = list(itertools.product(
        fixed_thresholds, min_areas, morph_kernels
    ))

    for idx, (thr, area, morph) in enumerate(combos, 1):
        result = evaluate(
            samples, thr, area, morph,
            blur_kernel, max_area, iou_threshold
        )
        records.append(result)

        print(
            "[%02d/30] thr=%2d min_area=%3d morph=%d "
            "-> P=%.4f R=%.4f F1=%.4f"
            % (
                idx, thr, area, morph,
                result["precision"],
                result["recall"],
                result["f1"],
            )
        )

    df = pd.DataFrame(records)
    ranked = df.sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    out_dir = ROOT / "outputs" / "opencv_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(
        out_dir / "val_grid_search.csv",
        index=False,
        encoding="utf-8-sig"
    )

    best = ranked.iloc[0]

    best_yaml = {
        "selection_basis": "validation_set_only",
        "selection_rule": "max_f1_then_recall_then_precision",
        "fixed_parameters": {
            "blur_kernel": blur_kernel,
            "max_area": max_area,
            "iou_threshold": iou_threshold,
            "use_otsu": False,
        },
        "best_parameters": {
            "fixed_threshold": int(best["fixed_threshold"]),
            "min_area": int(best["min_area"]),
            "morph_kernel": int(best["morph_kernel"]),
        },
        "validation_metrics": {
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
        },
    }

    with open(out_dir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_yaml, f, allow_unicode=True, sort_keys=False)

    print("\nTOP 10:")
    print(
        ranked.head(10)[[
            "fixed_threshold", "min_area", "morph_kernel",
            "tp", "fp", "fn", "precision", "recall", "f1"
        ]].to_string(index=False)
    )

    print("\n最佳参数：")
    print("fixed_threshold =", int(best["fixed_threshold"]))
    print("min_area        =", int(best["min_area"]))
    print("morph_kernel    =", int(best["morph_kernel"]))
    print("Precision       = %.4f" % best["precision"])
    print("Recall          = %.4f" % best["recall"])
    print("F1              = %.4f" % best["f1"])
    print("\n注意：仅使用 val 集选参，不根据 test 结果再次调参。")


if __name__ == "__main__":
    main()
