from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
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


def load_gt_original(path):
    rows = []

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue

        parsed = parse_annotation_line(line)
        if parsed is None:
            continue

        x1, y1, x2, y2, class_id_1based = parsed

        rows.append({
            "class_id": class_id_1based - 1,
            "box": [float(x1), float(y1), float(x2), float(y2)],
        })

    return rows


def get_class_name(class_names, class_id):
    if class_id in class_names:
        return str(class_names[class_id])

    key = str(class_id)
    if key in class_names:
        return str(class_names[key])

    return "class_%d" % int(class_id)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2, y2


def local_visual_metrics(gray, box):
    """
    计算三个简单的视觉属性：
    1. area_ratio：缺陷框面积 / 整图面积
    2. local_contrast：框内均值与周边环带均值的绝对差
    3. edge_density：扩展ROI中的 Canny 边缘像素比例
    """
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    box_area = float(bw * bh)
    area_ratio = box_area / float(w * h)

    inside = gray[y1:y2, x1:x2]

    pad_x = max(4, int(round(0.35 * bw)))
    pad_y = max(4, int(round(0.35 * bh)))

    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(w, x2 + pad_x)
    ey2 = min(h, y2 + pad_y)

    expanded = gray[ey1:ey2, ex1:ex2]

    ring_mask = np.ones(expanded.shape, dtype=np.uint8)

    rx1 = x1 - ex1
    ry1 = y1 - ey1
    rx2 = x2 - ex1
    ry2 = y2 - ey1

    ring_mask[ry1:ry2, rx1:rx2] = 0

    ring_values = expanded[ring_mask.astype(bool)]

    inside_mean = float(np.mean(inside)) if inside.size else 0.0

    if ring_values.size:
        ring_mean = float(np.mean(ring_values))
        local_contrast = abs(inside_mean - ring_mean)
    else:
        local_contrast = 0.0

    edges = cv2.Canny(expanded, 50, 150)
    edge_density = (
        float(np.count_nonzero(edges)) / float(edges.size)
        if edges.size
        else 0.0
    )

    return {
        "area_ratio": area_ratio,
        "local_contrast": local_contrast,
        "edge_density": edge_density,
    }


def build_reference_visual_thresholds(prepared_root):
    """
    使用整个 test 集 GT 的视觉属性建立参考阈值。
    这里只用于错误解释，不参与模型/参数调优。
    """
    image_dir = prepared_root / "images" / "test"
    gt_dir = prepared_root / "gt_original" / "test"

    area_values = []
    contrast_values = []
    edge_values = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        sid = image_path.stem
        gt_path = gt_dir / ("%s.txt" % sid)

        if not gt_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gt_rows = load_gt_original(gt_path)

        for gt in gt_rows:
            m = local_visual_metrics(gray, gt["box"])

            area_values.append(m["area_ratio"])
            contrast_values.append(m["local_contrast"])
            edge_values.append(m["edge_density"])

    if not area_values:
        raise RuntimeError("无法建立视觉参考阈值：test GT为空。")

    thresholds = {
        "small_area_ratio_q25": float(np.quantile(area_values, 0.25)),
        "low_contrast_q25": float(np.quantile(contrast_values, 0.25)),
        "complex_edge_density_q75": float(np.quantile(edge_values, 0.75)),
    }

    return thresholds


def match_quality_predictions(preds, gts, quality_threshold, match_iou):
    """
    质量工作点下执行 class-aware greedy match。
    返回：
      matched_pairs: list of dict
      unmatched_pred_indices
      unmatched_gt_indices
    """
    eligible = [
        (idx, pred)
        for idx, pred in enumerate(preds)
        if float(pred["conf"]) >= float(quality_threshold)
    ]

    eligible.sort(
        key=lambda item: float(item[1]["conf"]),
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()
    pairs = []

    for pred_idx, pred in eligible:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue

            if int(gt["class_id"]) != int(pred["class_id"]):
                continue

            iou = box_iou_xyxy(
                pred["box"],
                gt["box"],
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_gt_idx is not None
            and best_iou >= float(match_iou)
        ):
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

            pairs.append({
                "pred_idx": pred_idx,
                "gt_idx": best_gt_idx,
                "iou": float(best_iou),
            })

    quality_pred_indices = {
        idx
        for idx, pred in enumerate(preds)
        if float(pred["conf"]) >= float(quality_threshold)
    }

    unmatched_pred = sorted(
        quality_pred_indices - matched_pred
    )

    unmatched_gt = sorted(
        set(range(len(gts))) - matched_gt
    )

    return pairs, unmatched_pred, unmatched_gt


def nearest_prediction_info(gt, preds, quality_threshold, match_iou, loc_floor):
    """
    对一个 FN GT 做自动原因判定。
    """
    same_class = []
    wrong_class = []

    for idx, pred in enumerate(preds):
        iou = box_iou_xyxy(
            gt["box"],
            pred["box"],
        )

        item = {
            "pred_idx": idx,
            "iou": float(iou),
            "conf": float(pred["conf"]),
            "class_id": int(pred["class_id"]),
        }

        if int(pred["class_id"]) == int(gt["class_id"]):
            same_class.append(item)
        else:
            wrong_class.append(item)

    best_same = (
        max(same_class, key=lambda x: x["iou"])
        if same_class
        else None
    )

    best_wrong = (
        max(wrong_class, key=lambda x: x["iou"])
        if wrong_class
        else None
    )

    if (
        best_same is not None
        and best_same["iou"] >= float(match_iou)
        and best_same["conf"] < float(quality_threshold)
    ):
        return (
            "low_confidence_miss",
            best_same,
        )

    if (
        best_wrong is not None
        and best_wrong["iou"] >= float(match_iou)
    ):
        return (
            "class_confusion",
            best_wrong,
        )

    if (
        best_same is not None
        and float(loc_floor) <= best_same["iou"] < float(match_iou)
    ):
        return (
            "localization_iou",
            best_same,
        )

    return (
        "pure_miss",
        best_same if best_same is not None else best_wrong,
    )


def classify_fp_reason(pred_idx, pred, gts, matched_pairs, match_iou, loc_floor):
    matched_gt_indices = {
        int(x["gt_idx"])
        for x in matched_pairs
    }

    best_any = None

    for gt_idx, gt in enumerate(gts):
        iou = box_iou_xyxy(
            pred["box"],
            gt["box"],
        )

        item = {
            "gt_idx": gt_idx,
            "iou": float(iou),
            "gt_class_id": int(gt["class_id"]),
        }

        if best_any is None or item["iou"] > best_any["iou"]:
            best_any = item

    if best_any is None:
        return "background_false_positive", None

    same_class = (
        int(pred["class_id"])
        == int(best_any["gt_class_id"])
    )

    if (
        best_any["iou"] >= float(match_iou)
        and same_class
        and best_any["gt_idx"] in matched_gt_indices
    ):
        return "duplicate_prediction", best_any

    if (
        best_any["iou"] >= float(match_iou)
        and not same_class
    ):
        return "class_confusion", best_any

    if (
        same_class
        and float(loc_floor) <= best_any["iou"] < float(match_iou)
    ):
        return "localization_iou", best_any

    return "background_false_positive", best_any


def visual_factor_labels(metrics, thresholds):
    labels = []

    if metrics["area_ratio"] <= thresholds["small_area_ratio_q25"]:
        labels.append("small_defect")

    if metrics["local_contrast"] <= thresholds["low_contrast_q25"]:
        labels.append("low_contrast")

    if metrics["edge_density"] >= thresholds["complex_edge_density_q75"]:
        labels.append("complex_pattern")

    return labels


def draw_review_image(
    image,
    gts,
    preds,
    matched_pairs,
    unmatched_pred_indices,
    unmatched_gt_indices,
    quality_threshold,
    class_names,
):
    canvas = image.copy()

    matched_gt_indices = {
        int(x["gt_idx"])
        for x in matched_pairs
    }
    matched_pred_indices = {
        int(x["pred_idx"])
        for x in matched_pairs
    }

    # TP: green
    for pair in matched_pairs:
        gt = gts[pair["gt_idx"]]
        pred = preds[pair["pred_idx"]]

        x1, y1, x2, y2 = map(int, gt["box"])

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2,
        )

        label = "TP %s %.2f" % (
            get_class_name(class_names, pred["class_id"]),
            float(pred["conf"]),
        )

        cv2.putText(
            canvas,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    # FN: red GT
    for gt_idx in unmatched_gt_indices:
        gt = gts[gt_idx]
        x1, y1, x2, y2 = map(int, gt["box"])

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        label = "FN %s" % get_class_name(
            class_names,
            gt["class_id"],
        )

        cv2.putText(
            canvas,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # FP: blue prediction
    for pred_idx in unmatched_pred_indices:
        pred = preds[pred_idx]
        x1, y1, x2, y2 = map(int, pred["box"])

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 100, 0),
            2,
        )

        label = "FP %s %.2f" % (
            get_class_name(class_names, pred["class_id"]),
            float(pred["conf"]),
        )

        cv2.putText(
            canvas,
            label,
            (x1, min(canvas.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 100, 0),
            1,
            cv2.LINE_AA,
        )

    header = (
        "green=TP  red=FN  blue=FP  "
        "quality_conf=%.2f"
        % float(quality_threshold)
    )

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 24),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        header,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def main():
    cfg = load_config()

    prepared = resolve(
        cfg["dataset"]["prepared_root"]
    )

    quality_dir = (
        ROOT
        / "outputs"
        / "quality_analysis"
    )

    yolo_dir = (
        ROOT
        / "outputs"
        / "yolo"
    )

    top20_path = (
        quality_dir
        / "top_20_error_images.csv"
    )

    pred_path = (
        yolo_dir
        / "all_predictions_low_threshold.csv"
    )

    threshold_path = (
        yolo_dir
        / "selected_thresholds.csv"
    )

    if not top20_path.exists():
        raise FileNotFoundError(
            "未找到 %s。请先运行 05_quality_analysis.py。"
            % top20_path
        )

    if not pred_path.exists():
        raise FileNotFoundError(
            "未找到 %s。请先运行 04_evaluate_yolo.py。"
            % pred_path
        )

    if not threshold_path.exists():
        raise FileNotFoundError(
            "未找到 %s。请先运行 04_evaluate_yolo.py。"
            % threshold_path
        )

    top20 = pd.read_csv(top20_path)
    predictions_df = pd.read_csv(pred_path)
    threshold_df = pd.read_csv(threshold_path)

    quality_row = threshold_df[
        threshold_df["mode"] == "quality_high_recall"
    ]

    if len(quality_row) != 1:
        raise RuntimeError(
            "selected_thresholds.csv 中未唯一找到 "
            "quality_high_recall。"
        )

    quality_threshold = float(
        quality_row.iloc[0]["threshold"]
    )

    match_iou = float(
        cfg["evaluation"]["match_iou"]
    )

    low_prediction_threshold = float(
        cfg["evaluation"]["threshold_start"]
    )

    localization_iou_floor = min(
        0.10,
        max(0.05, match_iou * 0.30),
    )

    class_names = cfg["dataset"]["classes"]

    print("建立视觉参考阈值...")
    visual_thresholds = build_reference_visual_thresholds(
        prepared
    )

    print(
        "small area ratio <= %.6f"
        % visual_thresholds["small_area_ratio_q25"]
    )
    print(
        "low contrast <= %.3f"
        % visual_thresholds["low_contrast_q25"]
    )
    print(
        "complex edge density >= %.4f"
        % visual_thresholds["complex_edge_density_q75"]
    )
    print()

    out_dir = (
        ROOT
        / "outputs"
        / "error_review"
    )
    image_out_dir = (
        out_dir
        / "images"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    image_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    box_rows = []
    image_rows = []

    for rank, row in top20.reset_index(drop=True).iterrows():
        sid = str(row["sample_id"])

        image_path = (
            prepared
            / "images"
            / "test"
            / ("%s.jpg" % sid)
        )

        gt_path = (
            prepared
            / "gt_original"
            / "test"
            / ("%s.txt" % sid)
        )

        if not image_path.exists():
            raise FileNotFoundError(
                "缺少 test image：%s"
                % image_path
            )

        if not gt_path.exists():
            raise FileNotFoundError(
                "缺少 GT：%s"
                % gt_path
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                "无法读取图像：%s"
                % image_path
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gts = load_gt_original(
            gt_path
        )

        sub = predictions_df[
            predictions_df["sample_id"].astype(str) == sid
        ].copy()

        preds = []

        for _, pr in sub.iterrows():
            preds.append({
                "class_id": int(pr["class_id"]),
                "conf": float(pr["conf"]),
                "box": [
                    float(pr["x1"]),
                    float(pr["y1"]),
                    float(pr["x2"]),
                    float(pr["y2"]),
                ],
            })

        matched_pairs, unmatched_pred, unmatched_gt = (
            match_quality_predictions(
                preds,
                gts,
                quality_threshold,
                match_iou,
            )
        )

        reason_counter = Counter()
        class_counter = Counter()
        manual_flags = []

        # FN analysis
        for gt_idx in unmatched_gt:
            gt = gts[gt_idx]

            reason, nearest = nearest_prediction_info(
                gt,
                preds,
                quality_threshold,
                match_iou,
                localization_iou_floor,
            )

            metrics = local_visual_metrics(
                gray,
                gt["box"],
            )

            visual_factors = visual_factor_labels(
                metrics,
                visual_thresholds,
            )

            class_name = get_class_name(
                class_names,
                gt["class_id"],
            )

            reason_counter[reason] += 1
            class_counter[class_name] += 1

            manual_required = (
                reason == "pure_miss"
                and len(visual_factors) == 0
            )

            manual_flags.append(
                bool(manual_required)
            )

            box_rows.append({
                "sample_id": sid,
                "top20_rank": rank + 1,
                "error_side": "FN",
                "class_id": int(gt["class_id"]),
                "class_name": class_name,
                "primary_reason": reason,
                "secondary_visual_factors": ";".join(visual_factors),
                "gt_area_ratio": metrics["area_ratio"],
                "local_contrast": metrics["local_contrast"],
                "edge_density": metrics["edge_density"],
                "nearest_iou": (
                    nearest["iou"]
                    if nearest is not None
                    else np.nan
                ),
                "nearest_pred_conf": (
                    nearest["conf"]
                    if nearest is not None
                    else np.nan
                ),
                "nearest_pred_class_id": (
                    nearest["class_id"]
                    if nearest is not None
                    else np.nan
                ),
                "manual_check_required": manual_required,
                "possible_annotation_ambiguity": False,
            })

        # FP analysis
        for pred_idx in unmatched_pred:
            pred = preds[pred_idx]

            reason, nearest_gt = classify_fp_reason(
                pred_idx,
                pred,
                gts,
                matched_pairs,
                match_iou,
                localization_iou_floor,
            )

            reason_counter[reason] += 1

            class_name = get_class_name(
                class_names,
                pred["class_id"],
            )

            class_counter[class_name] += 1

            manual_required = (
                reason == "background_false_positive"
            )

            manual_flags.append(
                bool(manual_required)
            )

            box_rows.append({
                "sample_id": sid,
                "top20_rank": rank + 1,
                "error_side": "FP",
                "class_id": int(pred["class_id"]),
                "class_name": class_name,
                "primary_reason": reason,
                "secondary_visual_factors": "",
                "gt_area_ratio": np.nan,
                "local_contrast": np.nan,
                "edge_density": np.nan,
                "nearest_iou": (
                    nearest_gt["iou"]
                    if nearest_gt is not None
                    else np.nan
                ),
                "nearest_pred_conf": float(pred["conf"]),
                "nearest_pred_class_id": int(pred["class_id"]),
                "manual_check_required": manual_required,
                "possible_annotation_ambiguity": False,
            })

        distinct_reasons = sorted(
            reason_counter.keys()
        )

        if reason_counter:
            primary_reason = reason_counter.most_common(1)[0][0]
        else:
            primary_reason = "none"

        affected_classes = ";".join(
            x[0]
            for x in class_counter.most_common()
        )

        manual_required_image = (
            any(manual_flags)
            or len(distinct_reasons) >= 3
        )

        image_rows.append({
            "sample_id": sid,
            "top20_rank": rank + 1,
            "fp_expected_from_quality_csv": int(row["fp"]),
            "fn_expected_from_quality_csv": int(row["fn"]),
            "fp_reconstructed": int(len(unmatched_pred)),
            "fn_reconstructed": int(len(unmatched_gt)),
            "tp_reconstructed": int(len(matched_pairs)),
            "affected_classes": affected_classes,
            "primary_reason": primary_reason,
            "all_reasons": ";".join(distinct_reasons),
            "manual_check_required": bool(manual_required_image),
            "review_status": (
                "manual_check_recommended"
                if manual_required_image
                else "auto_review_sufficient"
            ),
        })

        canvas = draw_review_image(
            image,
            gts,
            preds,
            matched_pairs,
            unmatched_pred,
            unmatched_gt,
            quality_threshold,
            class_names,
        )

        cv2.imwrite(
            str(
                image_out_dir
                / (
                    "%02d_%s_review.jpg"
                    % (rank + 1, sid)
                )
            ),
            canvas,
        )

        print(
            "[%02d/20] %s -> TP=%d FP=%d FN=%d primary=%s"
            % (
                rank + 1,
                sid,
                len(matched_pairs),
                len(unmatched_pred),
                len(unmatched_gt),
                primary_reason,
            )
        )

    box_df = pd.DataFrame(box_rows)
    image_df = pd.DataFrame(image_rows)

    box_df.to_csv(
        out_dir / "top20_box_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    image_df.to_csv(
        out_dir / "automated_error_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reason_summary = (
        box_df.groupby(
            ["error_side", "primary_reason"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["error_side", "count"],
            ascending=[True, False],
        )
    )

    reason_summary.to_csv(
        out_dir / "error_reason_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_summary = (
        box_df.groupby(
            ["error_side", "class_name"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["error_side", "count"],
            ascending=[True, False],
        )
    )

    class_summary.to_csv(
        out_dir / "class_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manual_queue = image_df[
        image_df["manual_check_required"] == True
    ].copy()

    manual_queue.to_csv(
        out_dir / "manual_check_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_config = {
        "source": {
            "top20_file": str(top20_path),
            "prediction_file": str(pred_path),
            "selected_thresholds_file": str(threshold_path),
        },
        "frozen_evaluation": {
            "quality_threshold": quality_threshold,
            "low_prediction_threshold": low_prediction_threshold,
            "match_iou": match_iou,
            "localization_iou_floor": localization_iou_floor,
        },
        "visual_reference_thresholds": visual_thresholds,
        "reason_rules": {
            "low_confidence_miss": (
                "same-class prediction overlaps GT above match_iou "
                "but confidence is below quality threshold"
            ),
            "class_confusion": (
                "wrong-class prediction overlaps GT above match_iou"
            ),
            "localization_iou": (
                "same-class prediction is spatially near GT but "
                "IoU is below match_iou"
            ),
            "duplicate_prediction": (
                "extra same-class prediction overlaps an already matched GT"
            ),
            "background_false_positive": (
                "quality-threshold FP has no sufficient GT overlap"
            ),
            "pure_miss": (
                "FN has no adequate same-class/wrong-class prediction evidence"
            ),
        },
        "claim_boundary": (
            "Automated attribution is heuristic. "
            "The script does not automatically declare annotation errors. "
            "Rows flagged manual_check_required should be visually confirmed."
        ),
    }

    with open(
        out_dir / "review_rules_and_thresholds.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            audit_config,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print()
    print("=" * 72)
    print("自动错误审计完成")
    print("=" * 72)
    print("Top20 images:", len(image_df))
    print("Error events:", len(box_df))
    print("Manual-check images:", len(manual_queue))
    print()
    print("错误原因汇总：")
    print(reason_summary.to_string(index=False))
    print()
    print("输出目录：")
    print(out_dir)
    print()
    print(
        "说明：脚本不会自动判定 GT 标注错误；"
        "manual_check_required=True 的案例建议最后人工确认。"
    )


if __name__ == "__main__":
    main()
