from pathlib import Path
import random
import shutil
import yaml
import cv2
import pandas as pd
from common import load_config, resolve, set_seed

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


def find_split_ids(raw_root, filename):
    hits = list(raw_root.rglob(filename))
    if not hits:
        return []
    ids = []
    for line in hits[0].read_text(encoding="utf-8-sig").splitlines():
        s = line.strip().replace("\\", "/")
        if not s:
            continue
        name = Path(s).stem.replace("_test", "").replace("_temp", "")
        ids.append(name)
    return ids

def build_file_maps(raw_root):
    test_map = {p.stem.replace("_test", ""): p for p in raw_root.rglob("*_test.jpg")}
    temp_map = {p.stem.replace("_temp", ""): p for p in raw_root.rglob("*_temp.jpg")}
    ann_map = {
        p.stem: p
        for p in raw_root.rglob("*.txt")
        if p.name.lower() not in {"trainval.txt", "test.txt"}
    }
    return test_map, temp_map, ann_map

def convert_ann(ann, width, height):
    out = []
    for line in ann.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue

        parsed = parse_annotation_line(line)
        if parsed is None:
            continue

        x1, y1, x2, y2, c = parsed

        if c not in range(1, 7):
            raise ValueError(f"类别超出1-6：{ann} -> {line}")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"标注框越界：{ann} -> {line}")

        cls0 = c - 1
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        out.append(f"{cls0} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    return out

def main():
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    raw = resolve(cfg["dataset"]["raw_root"])
    prepared = resolve(cfg["dataset"]["prepared_root"])

    if prepared.exists():
        shutil.rmtree(prepared)

    for split in ["train", "val", "test"]:
        for sub in ["images", "labels", "templates", "gt_original"]:
            (prepared / sub / split).mkdir(parents=True, exist_ok=True)

    test_map, temp_map, ann_map = build_file_maps(raw)
    if not test_map:
        raise RuntimeError("没有找到 *_test.jpg。")

    trainval_ids = find_split_ids(raw, "trainval.txt")
    test_ids = find_split_ids(raw, "test.txt")

    if not trainval_ids or not test_ids:
        ids = sorted(test_map)
        random.Random(seed).shuffle(ids)
        n_test = round(len(ids) / 3)
        test_ids = ids[:n_test]
        trainval_ids = ids[n_test:]
        print("警告：未找到官方 split 文件，使用固定随机种子近似划分。")

    trainval_ids = [x for x in trainval_ids if x in test_map]
    test_ids = [x for x in test_ids if x in test_map]

    rng = random.Random(seed)
    rng.shuffle(trainval_ids)

    n_val = max(
        1,
        round(
            len(trainval_ids)
            * float(cfg["dataset"]["val_ratio_from_trainval"])
        ),
    )

    split_map = {x: "val" for x in trainval_ids[:n_val]}
    split_map.update({x: "train" for x in trainval_ids[n_val:]})
    split_map.update({x: "test" for x in test_ids})

    records, skipped = [], []

    for sid, split in sorted(split_map.items()):
        test_img = test_map.get(sid)
        temp_img = temp_map.get(sid)
        ann = ann_map.get(sid)

        if test_img is None or temp_img is None or ann is None:
            skipped.append({
                "sample_id": sid,
                "split": split,
                "has_test": test_img is not None,
                "has_template": temp_img is not None,
                "has_annotation": ann is not None,
            })
            continue

        im = cv2.imread(str(test_img))
        if im is None:
            skipped.append({
                "sample_id": sid,
                "split": split,
                "reason": "cannot_read_test_image",
            })
            continue

        h, w = im.shape[:2]
        labels = convert_ann(ann, w, h)

        shutil.copy2(
            test_img,
            prepared / "images" / split / f"{sid}.jpg",
        )
        shutil.copy2(
            temp_img,
            prepared / "templates" / split / f"{sid}.jpg",
        )
        shutil.copy2(
            ann,
            prepared / "gt_original" / split / f"{sid}.txt",
        )

        (prepared / "labels" / split / f"{sid}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""),
            encoding="utf-8",
        )

        records.append({
            "sample_id": sid,
            "split": split,
            "boxes": len(labels),
            "width": w,
            "height": h,
        })

    names = {i: cfg["dataset"]["classes"][i] for i in range(6)}
    data_yaml = {
        "path": str(prepared.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }

    with open(prepared / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data_yaml,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    df = pd.DataFrame(records)
    df.to_csv(
        prepared / "split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(skipped).to_csv(
        prepared / "skipped_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("YOLO 数据集准备完成。")
    print()
    print(
        df.groupby("split").agg(
            images=("sample_id", "count"),
            boxes=("boxes", "sum"),
        )
    )
    print()
    print(f"跳过样本：{len(skipped)}")

if __name__ == "__main__":
    main()
