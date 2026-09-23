from pathlib import Path
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


def detect_candidates(
    test_bgr,
    template_bgr,
    fixed_threshold,
    min_area,
    morph_kernel,
    blur_kernel,
    max_area,
):
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    blur_kernel = int(blur_kernel)
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        test_gray = cv2.GaussianBlur(
            test_gray,
            (blur_kernel, blur_kernel),
            0,
        )
        template_gray = cv2.GaussianBlur(
            template_gray,
            (blur_kernel, blur_kernel),
            0,
        )

    diff = cv2.absdiff(test_gray, template_gray)

    _, mask = cv2.threshold(
        diff,
        int(fixed_threshold),
        255,
        cv2.THRESH_BINARY,
    )

    morph_kernel = int(morph_kernel)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (morph_kernel, morph_kernel),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if float(min_area) <= area <= float(max_area):
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))

    return boxes, mask, diff


def greedy_match(pred_boxes, gt_boxes, iou_threshold):
    used_gt = set()
    tp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = None

        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt:
                continue

            iou = box_iou_xyxy(
                pred,
                gt[:4],
            )

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if (
            best_idx is not None
            and best_iou >= float(iou_threshold)
        ):
            used_gt.add(best_idx)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def main():
    cfg = load_config()

    prepared = resolve(
        cfg["dataset"]["prepared_root"]
    )

    best_path = (
        ROOT
        / "outputs"
        / "opencv_tuning"
        / "best_params.yaml"
    )

    if not best_path.exists():
        raise FileNotFoundError(
            "未找到 outputs/opencv_tuning/best_params.yaml。"
            "请先运行 02a_tune_opencv_val.py。"
        )

    with open(
        best_path,
        "r",
        encoding="utf-8",
    ) as f:
        best_cfg = yaml.safe_load(f)

    bp = best_cfg["best_parameters"]
    fp_cfg = best_cfg["fixed_parameters"]

    fixed_threshold = int(
        bp["fixed_threshold"]
    )
    min_area = int(
        bp["min_area"]
    )
    morph_kernel = int(
        bp["morph_kernel"]
    )

    blur_kernel = int(
        fp_cfg["blur_kernel"]
    )
    max_area = int(
        fp_cfg["max_area"]
    )
    iou_threshold = float(
        fp_cfg["iou_threshold"]
    )

    print("冻结参数：")
    print(
        "fixed_threshold = %d"
        % fixed_threshold
    )
    print(
        "min_area        = %d"
        % min_area
    )
    print(
        "morph_kernel    = %d"
        % morph_kernel
    )
    print(
        "blur_kernel     = %d"
        % blur_kernel
    )
    print(
        "max_area        = %d"
        % max_area
    )
    print(
        "IoU threshold   = %.2f"
        % iou_threshold
    )
    print()

    test_dir = (
        prepared
        / "images"
        / "test"
    )
    template_dir = (
        prepared
        / "templates"
        / "test"
    )
    gt_dir = (
        prepared
        / "gt_original"
        / "test"
    )

    test_paths = sorted(
        test_dir.glob("*.jpg")
    )

    if not test_paths:
        raise RuntimeError(
            "test 集为空。"
        )

    out = (
        ROOT
        / "outputs"
        / "opencv_frozen_test"
    )
    vis = out / "visualizations"

    out.mkdir(
        parents=True,
        exist_ok=True,
    )
    vis.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    TP = 0
    FP = 0
    FN = 0

    for idx, test_path in enumerate(
        test_paths
    ):
        sid = test_path.stem

        template_path = (
            template_dir
            / ("%s.jpg" % sid)
        )
        gt_path = (
            gt_dir
            / ("%s.txt" % sid)
        )

        if not template_path.exists():
            raise FileNotFoundError(
                "缺少模板：%s"
                % template_path
            )

        if not gt_path.exists():
            raise FileNotFoundError(
                "缺少标注：%s"
                % gt_path
            )

        test = cv2.imread(
            str(test_path)
        )
        template = cv2.imread(
            str(template_path)
        )

        if (
            test is None
            or template is None
        ):
            raise RuntimeError(
                "图像读取失败：%s"
                % sid
            )

        gt = load_gt(
            gt_path
        )

        preds, mask, diff = (
            detect_candidates(
                test,
                template,
                fixed_threshold,
                min_area,
                morph_kernel,
                blur_kernel,
                max_area,
            )
        )

        tp, fp, fn = (
            greedy_match(
                preds,
                gt,
                iou_threshold,
            )
        )

        TP += tp
        FP += fp
        FN += fn

        rows.append({
            "sample_id": sid,
            "gt_boxes": len(gt),
            "pred_boxes": len(preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        # 只保存前20个样本的可视化，控制输出体积
        if idx < 20:
            canvas = test.copy()

            for x1, y1, x2, y2, *_ in gt:
                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1,
                )

            for x1, y1, x2, y2 in preds:
                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    1,
                )

            cv2.imwrite(
                str(
                    vis
                    / ("%s_boxes.jpg" % sid)
                ),
                canvas,
            )

            cv2.imwrite(
                str(
                    vis
                    / ("%s_mask.png" % sid)
                ),
                mask,
            )

            cv2.imwrite(
                str(
                    vis
                    / ("%s_diff.png" % sid)
                ),
                diff,
            )

    precision = (
        TP / float(TP + FP)
        if TP + FP
        else 0.0
    )

    recall = (
        TP / float(TP + FN)
        if TP + FN
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        out / "image_level_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = pd.DataFrame([{
        "selection_source": "validation_set",
        "fixed_threshold": fixed_threshold,
        "min_area": min_area,
        "morph_kernel": morph_kernel,
        "blur_kernel": blur_kernel,
        "max_area": max_area,
        "iou_threshold": iou_threshold,
        "test_images": len(test_paths),
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "note": (
            "OpenCV frozen test. "
            "Parameters selected only on validation set."
        ),
    }])

    metrics.to_csv(
        out / "frozen_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "处理测试图：%d"
        % len(test_paths)
    )
    print(
        "TP=%d, FP=%d, FN=%d"
        % (TP, FP, FN)
    )
    print(
        "OpenCV frozen test: "
        "P=%.4f, R=%.4f, F1=%.4f"
        % (
            precision,
            recall,
            f1,
        )
    )
    print()
    print(
        "输出目录：%s"
        % out
    )
    print()
    print(
        "该结果为冻结后的正式 test 结果。"
        "不要再根据 test 结果调整 OpenCV 参数。"
    )


if __name__ == "__main__":
    main()
