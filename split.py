"""
Split UCF101 dataset into train/test sets using the official split files.

Directory structure expected:
    UCF101/
        <ClassName>/
            <video>.avi
    ucfTrainTestlist/
        trainlist01.txt
        trainlist02.txt
        trainlist03.txt
        testlist01.txt
        testlist02.txt
        testlist03.txt

Output structure:
    output/
        split1/
            train/  (symlinks or copies)
            test/
        split2/
            train/
            test/
        split3/
            train/
            test/
"""

import os
import shutil
import argparse
from pathlib import Path


def parse_trainlist(filepath):
    """Parse a trainlist file. Lines are: ClassName/video.avi <label>"""
    videos = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                videos.append(parts[0])  # e.g. "Archery/v_Archery_g01_c01.avi"
    return videos


def parse_testlist(filepath):
    """Parse a testlist file. Lines are: ClassName/video.avi (no label)"""
    videos = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                videos.append(line)
    return videos


def organize_split(split_num, ucf_dir, split_dir, output_dir, use_symlinks=True):
    train_file = split_dir / f"trainlist0{split_num}.txt"
    test_file  = split_dir / f"testlist0{split_num}.txt"

    train_videos = parse_trainlist(train_file)
    test_videos  = parse_testlist(test_file)

    split_out = output_dir / f"split{split_num}"

    for subset, videos in [("train", train_videos), ("test", test_videos)]:
        subset_dir = split_out / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        missing = 0
        for rel_path in videos:
            src = ucf_dir / rel_path
            class_name = Path(rel_path).parent.name
            dst_class = subset_dir / class_name
            dst_class.mkdir(exist_ok=True)
            dst = dst_class / Path(rel_path).name

            if not src.exists():
                missing += 1
                continue

            if dst.exists():
                continue

            if use_symlinks:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        print(f"  Split {split_num} | {subset:5s}: {len(videos)} videos listed"
              + (f", {missing} missing" if missing else ""))


def main():
    parser = argparse.ArgumentParser(description="Organise UCF101 into train/test splits.")
    parser.add_argument("--ucf_dir",   required=True, help="Path to UCF-101 video folder")
    parser.add_argument("--split_dir", required=True, help="Path to ucfTrainTestlist folder")
    parser.add_argument("--output_dir",required=True, help="Where to write the organised splits")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks (uses more disk space)")
    args = parser.parse_args()

    ucf_dir   = Path(args.ucf_dir)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    assert ucf_dir.exists(),   f"UCF dir not found: {ucf_dir}"
    assert split_dir.exists(), f"Split dir not found: {split_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)
    use_symlinks = not args.copy

    print(f"UCF-101 dir : {ucf_dir}")
    print(f"Split files : {split_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Mode        : {'copy' if args.copy else 'symlink'}\n")

    for split_num in range(1, 4):
        organize_split(split_num, ucf_dir, split_dir, output_dir, use_symlinks)

    print("\nDone.")


if __name__ == "__main__":
    main()