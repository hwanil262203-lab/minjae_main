#!/usr/bin/env python3
"""
여러 개의 YOLO 형식 데이터셋(Roboflow "YOLOv8" export)을 하나로 합치고
클래스 이름을 통일(재매핑)하는 스크립트.

사용 예:
  # 1) 어떤 클래스가 있고 어떻게 매핑되는지 먼저 확인 (파일 복사 없음)
  python tools/merge_yolo_datasets.py --src datasets/obstacle --src datasets/safewalkbd --out datasets/merged --dry-run

  # 2) 실제 병합
  python tools/merge_yolo_datasets.py --src datasets/obstacle --src datasets/safewalkbd --out datasets/merged

  # 매핑을 직접 추가/수정하고 싶을 때 (원본이름=대상클래스, 대소문자 무시, 대상을 비우면 제외)
  python tools/merge_yolo_datasets.py ... --map "Over-bridge=stairs" --map "Tree="

  # 대상 클래스 목록을 바꾸고 싶을 때
  python tools/merge_yolo_datasets.py ... --classes crosswalk stairs pole

매핑되지 않은 클래스의 라벨은 버려진다. 라벨이 하나도 남지 않은 이미지는
--keep-background 비율만큼만 '배경 이미지'로 남기고 나머지는 제외한다.
"""
import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 최종 클래스 (순서 = 클래스 ID). 앱의 '횡단보도/계단/기둥,가로등' + 보행 위험물 2종
DEFAULT_CLASSES = ["crosswalk", "stairs", "pole", "person", "vehicle"]

# 원본 클래스 이름(정규화 후) -> 대상 클래스. 정규화: 소문자, '-'/'_' -> 공백, 공백 정리
DEFAULT_ALIASES = {
    # crosswalk
    "crosswalk": "crosswalk", "crosswalks": "crosswalk", "cross walk": "crosswalk",
    "zebra crossing": "crosswalk", "zebra cross": "crosswalk", "pedestrian crossing": "crosswalk",
    "crossing": "crosswalk",
    # stairs
    "stairs": "stairs", "stair": "stairs", "staircase": "stairs", "stairs up": "stairs",
    "stairs down": "stairs", "stairs upward": "stairs", "stairs downward": "stairs",
    "upstairs": "stairs", "downstairs": "stairs", "step": "stairs", "steps": "stairs",
    "upstair": "stairs", "downstair": "stairs", "ascending stairs": "stairs", "descending stairs": "stairs",
    "escaliers": "stairs",
    # pole
    "pole": "pole", "poles": "pole", "lamp post": "pole", "lamppost": "pole", "lamp": "pole",
    "street light": "pole", "streetlight": "pole", "light pole": "pole", "electric pole": "pole",
    "utility pole": "pole", "pillar": "pole", "column": "pole", "post": "pole", "sign post": "pole",
    # person
    "person": "person", "people": "person", "pedestrian": "person", "human": "person", "man": "person",
    # vehicle
    "vehicle": "vehicle", "vehicles": "vehicle", "car": "vehicle", "cars": "vehicle", "bus": "vehicle",
    "truck": "vehicle", "van": "vehicle", "motorcycle": "vehicle", "motorbike": "vehicle",
    "bike": "vehicle", "bicycle": "vehicle", "tram": "vehicle", "tramway": "vehicle",
    "train": "vehicle", "rickshaw": "vehicle", "auto rickshaw": "vehicle", "cng": "vehicle",
    "scooter": "vehicle",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]


def normalize(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[-_]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_names(yaml_path: Path):
    """data.yaml 의 names 를 [id -> name] 리스트로 읽는다 (PyYAML 없이도 동작)."""
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        return [str(n) for n in names]
    except ImportError:
        pass
    text = yaml_path.read_text(encoding="utf-8")
    m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if m:
        return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    names = []
    in_names = False
    for line in text.splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            mm = re.match(r"\s*(?:-\s*|\d+:\s*)(.+)$", line)
            if mm and (line.startswith(" ") or line.startswith("-")):
                names.append(mm.group(1).strip().strip("'\""))
            else:
                break
    if not names:
        sys.exit(f"names 를 읽지 못했습니다: {yaml_path}")
    return names


def find_split_dir(src: Path, split: str):
    """src/<split>/images 를 찾는다. Roboflow export 기본 구조."""
    cand = src / split / "images"
    if cand.is_dir():
        return cand
    # 일부 export 는 'val' 을 쓴다
    if split == "valid":
        cand = src / "val" / "images"
        if cand.is_dir():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, help="YOLO 데이터셋 폴더 (data.yaml 포함). 여러 번 지정")
    ap.add_argument("--out", required=True, help="병합 결과 폴더")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="최종 클래스 목록(순서=ID)")
    ap.add_argument("--map", action="append", default=[], help='추가 매핑 "원본이름=대상" (대상을 비우면 제외)')
    ap.add_argument("--keep-background", type=float, default=0.05, help="라벨 없는 이미지를 남길 비율 (0~1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="복사하지 않고 매핑/통계만 출력")
    args = ap.parse_args()

    random.seed(args.seed)
    targets = args.classes
    target_id = {t: i for i, t in enumerate(targets)}
    aliases = {k: v for k, v in DEFAULT_ALIASES.items() if v in target_id}
    for item in args.map:
        if "=" not in item:
            sys.exit(f"--map 형식 오류: {item}")
        k, v = item.split("=", 1)
        v = v.strip()
        if v and v not in target_id:
            sys.exit(f"--map 대상 '{v}' 가 --classes 에 없습니다")
        aliases[normalize(k)] = v or None

    out = Path(args.out)
    srcs = [Path(s) for s in args.src]

    # ---- 1. 각 소스의 클래스와 매핑 출력 ----
    per_src_map = {}  # src -> {orig_id: target_id or None}
    print("=" * 70)
    print("클래스 매핑")
    print("=" * 70)
    for src in srcs:
        yaml_path = src / "data.yaml"
        if not yaml_path.exists():
            sys.exit(f"data.yaml 없음: {yaml_path}")
        names = load_names(yaml_path)
        mapping = {}
        print(f"\n[{src}]  ({len(names)} classes)")
        for i, n in enumerate(names):
            t = aliases.get(normalize(n))
            mapping[i] = target_id[t] if t else None
            mark = f"-> {t}" if t else "-> (제외)"
            print(f"  {i:3d}  {n:<25} {mark}")
        per_src_map[src] = mapping

    # ---- 2. 이미지/라벨 순회 ----
    stats = defaultdict(Counter)      # split -> Counter(target name -> n boxes)
    img_stats = defaultdict(Counter)  # split -> Counter(kept / dropped / background)
    plan = []                         # (split, src_img, label_text, dst_stem)

    for src in srcs:
        mapping = per_src_map[src]
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", src.name).strip("_") or "ds"
        for split in SPLITS:
            img_dir = find_split_dir(src, split)
            if img_dir is None:
                continue
            lbl_dir = img_dir.parent / "labels"
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                kept_lines = []
                if lbl.exists():
                    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            cid = int(float(parts[0]))
                        except ValueError:
                            continue
                        tid = mapping.get(cid)
                        if tid is None:
                            continue
                        kept_lines.append(" ".join([str(tid)] + parts[1:]))
                        stats[split][targets[tid]] += 1
                if kept_lines:
                    img_stats[split]["kept"] += 1
                    plan.append((split, img, "\n".join(kept_lines) + "\n", f"{prefix}_{img.stem}"))
                elif random.random() < args.keep_background:
                    img_stats[split]["background"] += 1
                    plan.append((split, img, "", f"{prefix}_{img.stem}"))
                else:
                    img_stats[split]["dropped"] += 1

    # ---- 3. 통계 ----
    print("\n" + "=" * 70)
    print("병합 결과 통계")
    print("=" * 70)
    for split in SPLITS:
        if not img_stats[split] and not stats[split]:
            continue
        s = img_stats[split]
        print(f"\n{split}: 이미지 {s['kept']} (라벨 있음) + {s['background']} (배경) / 제외 {s['dropped']}")
        for t in targets:
            print(f"   {t:<12} {stats[split][t]:7d} boxes")

    missing = [t for t in targets if sum(stats[sp][t] for sp in SPLITS) == 0]
    if missing:
        print(f"\n[주의] 박스가 하나도 없는 클래스: {missing}  (--map 으로 원본 이름을 연결하거나 --classes 에서 빼세요)")

    if args.dry_run:
        print("\n(dry-run) 파일을 복사하지 않았습니다.")
        return

    # ---- 4. 복사 ----
    if out.exists() and any(out.iterdir()):
        sys.exit(f"출력 폴더가 비어 있지 않습니다: {out}")
    for split, img, lbl_text, stem in plan:
        dst_img = out / split / "images" / (stem + img.suffix.lower())
        dst_lbl = out / split / "labels" / (stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)
        dst_lbl.write_text(lbl_text, encoding="utf-8")

    # valid 가 없으면 train 을 val 로도 쓰도록 처리
    val_dir = "valid/images" if (out / "valid" / "images").exists() else "train/images"
    test_line = "test: test/images\n" if (out / "test" / "images").exists() else ""
    names_yaml = "\n".join(f"  {i}: {t}" for i, t in enumerate(targets))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: {val_dir}\n"
        f"{test_line}"
        f"nc: {len(targets)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
    print(f"\n완료: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
