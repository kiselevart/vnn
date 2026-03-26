#!/usr/bin/env python3
"""
Audit preprocessed video dataset splits for duplication and leakage.

Usage:
    python tools/audit_splits.py --pre_dir /path/to/preprocessed
    python tools/audit_splits.py  # uses mypath.py resolution for all known datasets
"""

import os
import sys
import argparse
from collections import defaultdict


def audit_dir(pre_dir: str, dataset_name: str = ""):
    label = dataset_name or os.path.basename(pre_dir)
    splits = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(pre_dir, s))]

    if not splits:
        print(f"[{label}] No train/val/test subdirs found in {pre_dir} — skipping.")
        return

    # Build: video_name -> [(split, class)]
    vid_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_sizes: dict[str, int] = {}

    for split in splits:
        count = 0
        split_dir = os.path.join(pre_dir, split)
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for vid in os.listdir(cls_dir):
                if os.path.isdir(os.path.join(cls_dir, vid)):
                    vid_locations[vid].append((split, cls))
                    count += 1
        split_sizes[split] = count

    print(f"\n{'='*60}")
    print(f"Dataset : {label}")
    print(f"Path    : {pre_dir}")
    for s in splits:
        print(f"  {s:6s}: {split_sizes[s]} videos")

    # --- 1. Cross-split duplicates (same video name in multiple splits) ---
    cross_split = {
        v: locs for v, locs in vid_locations.items()
        if len({s for s, c in locs}) > 1
    }

    # --- 2. Label conflicts (same video, different class in different splits) ---
    label_conflicts = {
        v: locs for v, locs in cross_split.items()
        if len({c for s, c in locs}) > 1
    }

    # --- 3. Within-split duplicates (same video name in multiple classes, same split) ---
    within_split: dict[str, list[tuple[str, str]]] = {}
    for vid, locs in vid_locations.items():
        by_split: dict[str, list[str]] = defaultdict(list)
        for s, c in locs:
            by_split[s].append(c)
        for s, classes in by_split.items():
            if len(classes) > 1:
                within_split.setdefault(vid, []).extend([(s, c) for c in classes])

    # --- 4. Class coverage: check every class present in train is also in val ---
    split_classes: dict[str, set[str]] = {}
    for split in splits:
        split_dir = os.path.join(pre_dir, split)
        split_classes[split] = {
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        }

    if "train" in split_classes and "val" in split_classes:
        train_only = split_classes["train"] - split_classes["val"]
        val_only = split_classes["val"] - split_classes["train"]
    else:
        train_only = val_only = set()

    # --- Report ---
    ok = True

    if cross_split:
        ok = False
        print(f"\n[FAIL] Cross-split duplicates: {len(cross_split)} video names appear in multiple splits")
        for vid, locs in sorted(cross_split.items())[:10]:
            print(f"  {vid}:")
            for s, c in sorted(locs):
                print(f"    {s}/{c}")
        if len(cross_split) > 10:
            print(f"  ... and {len(cross_split) - 10} more")
    else:
        print("\n[OK] No cross-split duplicates")

    if label_conflicts:
        ok = False
        print(f"\n[FAIL] Label conflicts: {len(label_conflicts)} videos have different class labels across splits")
        for vid, locs in sorted(label_conflicts.items())[:5]:
            classes = sorted({c for s, c in locs})
            print(f"  {vid} — labeled as: {classes}")
    else:
        print("[OK] No label conflicts")

    if within_split:
        ok = False
        print(f"\n[FAIL] Within-split duplicates: {len(within_split)} video names appear in multiple classes in the same split")
        for vid, locs in sorted(within_split.items())[:5]:
            print(f"  {vid}: {sorted(set(f'{s}/{c}' for s,c in locs))}")
    else:
        print("[OK] No within-split class duplicates")

    if train_only:
        print(f"\n[WARN] {len(train_only)} class(es) in train but NOT in val: {sorted(train_only)[:5]}")
    if val_only:
        print(f"[WARN] {len(val_only)} class(es) in val but NOT in train: {sorted(val_only)[:5]}")

    if ok and not train_only and not val_only:
        print("\n[PASS] All checks passed — no leakage or duplication detected.")
    elif ok:
        print("\n[PASS with warnings] No leakage/duplication, but class coverage mismatch exists.")


def main():
    parser = argparse.ArgumentParser(description="Audit preprocessed video dataset splits.")
    parser.add_argument("--pre_dir", type=str, default=None,
                        help="Path to preprocessed dataset root (must contain train/val/test subdirs).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for mypath.py resolution (e.g. ucf101, hmdb51, ucf10, ucf11). "
                             "Used when --pre_dir is not given.")
    args = parser.parse_args()

    if args.pre_dir:
        audit_dir(args.pre_dir)
    elif args.dataset:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mypath import Path
        _, pre_dir = Path.db_dir(args.dataset)
        audit_dir(pre_dir, args.dataset)
    else:
        # Audit all known datasets
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mypath import Path
        for db in ("ucf101", "hmdb51", "ucf10", "ucf11"):
            try:
                _, pre_dir = Path.db_dir(db)
                if os.path.exists(pre_dir):
                    audit_dir(pre_dir, db)
                else:
                    print(f"\n[SKIP] {db}: preprocessed dir not found ({pre_dir})")
            except NotImplementedError:
                pass


if __name__ == "__main__":
    main()
