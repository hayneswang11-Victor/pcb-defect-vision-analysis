from pathlib import Path
from collections import Counter
import pandas as pd
import cv2
from common import ROOT, load_config, resolve

def parse_annotation_line(line):
    """
    DeepPCB 官方标注实际采用空白分隔：
        x1 y1 x2 y2 class_id
    为兼容其他镜像/转换版本，同时接受逗号分隔。
    """
    line = line.strip()
    if not line:
        return None
    parts = line.replace(",", " ").split()
    if len(parts) != 5:
        raise ValueError(f"bad_annotation_field_count: {line}")
    x1, y1, x2, y2, c = map(int, parts)
    return x1, y1, x2, y2, c


def build_file_maps(raw_root):
    test_map = {p.stem.replace("_test", ""): p for p in raw_root.rglob("*_test.jpg")}
    temp_map = {p.stem.replace("_temp", ""): p for p in raw_root.rglob("*_temp.jpg")}
    ann_map = {
        p.stem: p
        for p in raw_root.rglob("*.txt")
        if p.name.lower() not in {"trainval.txt", "test.txt"}
    }
    return test_map, temp_map, ann_map

def main():
    cfg = load_config()
    raw_root = resolve(cfg["dataset"]["raw_root"])
    if not raw_root.exists():
        raise FileNotFoundError(
            f"未找到 {raw_root}。请确认 DeepPCB 已放在 data/raw/DeepPCB。"
        )

    test_map, temp_map, ann_map = build_file_maps(raw_root)
    rows, issues, counts = [], [], Counter()

    for sid, test_img in sorted(test_map.items()):
        temp_img = temp_map.get(sid)
        ann = ann_map.get(sid)

        if temp_img is None:
            issues.append({"sample_id": sid, "reason": "missing_template"})
            continue
        if ann is None:
            issues.append({"sample_id": sid, "reason": "missing_annotation"})
            continue

        im = cv2.imread(str(test_img))
        temp_im = cv2.imread(str(temp_img))
        if im is None:
            issues.append({"sample_id": sid, "reason": "cannot_read_test_image"})
            continue
        if temp_im is None:
            issues.append({"sample_id": sid, "reason": "cannot_read_template_image"})
            continue

        h, w = im.shape[:2]
        th, tw = temp_im.shape[:2]
        if (h, w) != (th, tw):
            issues.append({
                "sample_id": sid,
                "reason": "test_template_shape_mismatch",
                "test_shape": f"{w}x{h}",
                "template_shape": f"{tw}x{th}",
            })
            continue

        n = 0
        for line in ann.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                parsed = parse_annotation_line(line)
            except Exception as exc:
                issues.append({
                    "sample_id": sid,
                    "reason": str(exc),
                    "line": line,
                })
                continue

            if parsed is None:
                continue
            x1, y1, x2, y2, c = parsed

            if c not in range(1, 7):
                issues.append({
                    "sample_id": sid,
                    "reason": "invalid_class",
                    "line": line,
                })
                continue
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                issues.append({
                    "sample_id": sid,
                    "reason": "invalid_box",
                    "line": line,
                })
                continue

            counts[c] += 1
            n += 1

        rows.append({
            "sample_id": sid,
            "width": w,
            "height": h,
            "boxes": n,
            "test_image": str(test_img.relative_to(raw_root)),
            "template_image": str(temp_img.relative_to(raw_root)),
            "annotation": str(ann.relative_to(raw_root)),
        })

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    pd.DataFrame(rows).to_csv(
        out / "raw_dataset_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(issues).to_csv(
        out / "raw_dataset_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    names = cfg["dataset"]["classes"]
    summary = [
        {"class_id": i - 1, "class_name": names[i - 1], "boxes": counts[i]}
        for i in range(1, 7)
    ]
    pd.DataFrame(summary).to_csv(
        out / "raw_class_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    extra_templates = sorted(set(temp_map) - set(test_map))
    extra_annotations = sorted(set(ann_map) - set(test_map))

    pd.DataFrame(
        {"extra_template_sample_id": pd.Series(extra_templates, dtype="string")}
    ).to_csv(
        out / "extra_templates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {"extra_annotation_sample_id": pd.Series(extra_annotations, dtype="string")}
    ).to_csv(
        out / "extra_annotations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"测试图像：{len(test_map)}")
    print(f"模板图像：{len(temp_map)}")
    print(f"标注文件：{len(ann_map)}")
    print(f"有效样本：{len(rows)}")
    print(f"问题记录：{len(issues)}")
    print(f"额外模板：{len(extra_templates)}")
    print(f"额外标注：{len(extra_annotations)}")
    print()
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
