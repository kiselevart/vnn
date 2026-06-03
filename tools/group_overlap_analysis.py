"""
Group-level contamination analysis for UCF-101 and HMDB-51.

Compares two evaluation protocols:
  - Original (buggy): random 80/20 clip-level split (train_test_split, seed=42)
  - Corrected: UCF-101 official split files / HMDB-51 official split files

A test clip is "contaminated" if its source group also appears in the training
split. For UCF-101, the group is encoded in the filename as gXX in
v_Class_gXX_cYY.avi. For HMDB-51, the group is the source-video identifier
(filename with trailing clip index stripped).

Usage (run from vnn/):
    python3 tools/group_overlap_analysis.py

Requires:
  - HMDB-51: data/HMDB51/ with class subdirectories and testTrainMulti_7030_splits/
  - UCF-101: data/UCF-101/ with ucfTrainTestlist/trainlist0X.txt + testlist0X.txt
             and class subdirectories containing .avi files

UCF-101 data is only available on the training server. HMDB-51 can run locally.
"""

import os
import re
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ucf101_group(filename):
    """Extract actor group ID from a UCF-101 filename.
    e.g. v_Basketball_g01_c03.avi -> 'g01'
    """
    m = re.search(r'_(g\d+)_c\d+', filename)
    return m.group(1) if m else 'g00'


def hmdb51_source_group(clip_name):
    """Strip trailing clip index from an HMDB-51 clip name.
    e.g. April_09_brush_hair_u_nm_np1_ba_goo_2 -> April_09_brush_hair_u_nm_np1_ba_goo
    """
    return re.sub(r'_\d+$', '', clip_name)


# ---------------------------------------------------------------------------
# UCF-101
# ---------------------------------------------------------------------------

def analyze_ucf101_buggy(data_root):
    """Simulate the original random split on raw UCF-101 class directories."""
    total_test = 0
    contaminated_test = 0
    total_groups = 0
    contaminated_groups = 0

    for cls in sorted(os.listdir(data_root)):
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        video_files = sorted(
            f for f in os.listdir(cls_dir) if f.lower().endswith('.avi')
        )
        if len(video_files) < 3:
            continue

        _, test = train_test_split(video_files, test_size=0.2, random_state=42)
        train_val, _ = train_test_split(video_files, test_size=0.2, random_state=42)
        train, _ = train_test_split(train_val, test_size=0.2, random_state=42)

        train_groups = set(ucf101_group(v) for v in train)
        cls_test_groups = set()
        for v in test:
            g = ucf101_group(v)
            total_test += 1
            cls_test_groups.add(g)
            if g in train_groups:
                contaminated_test += 1

        for g in cls_test_groups:
            total_groups += 1
            if g in train_groups:
                contaminated_groups += 1

    return total_test, contaminated_test, total_groups, contaminated_groups


def analyze_ucf101_official(ucf_root, split_idx):
    """Compute group overlap for the official UCF-101 split files."""
    list_dir = os.path.join(ucf_root, 'ucfTrainTestlist')
    train_file = os.path.join(list_dir, f'trainlist0{split_idx}.txt')
    test_file  = os.path.join(list_dir, f'testlist0{split_idx}.txt')

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        return None

    def parse(path):
        vids = []
        with open(path) as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    fname = tokens[0].split('/')[-1]  # strip class/ prefix
                    vids.append(fname)
        return vids

    train_vids = parse(train_file)
    test_vids  = parse(test_file)

    train_groups = set(ucf101_group(v) for v in train_vids)
    contaminated = sum(1 for v in test_vids if ucf101_group(v) in train_groups)
    return len(test_vids), contaminated


# ---------------------------------------------------------------------------
# HMDB-51
# ---------------------------------------------------------------------------

def analyze_hmdb51_buggy(hmdb_root):
    """Simulate the original random split on local HMDB-51 frame directories."""
    classes = sorted(
        d for d in os.listdir(hmdb_root)
        if os.path.isdir(os.path.join(hmdb_root, d))
        and d != 'testTrainMulti_7030_splits'
    )

    total_test = 0
    contaminated_test = 0
    total_groups = 0
    contaminated_groups = 0

    for cls in classes:
        cls_dir = os.path.join(hmdb_root, cls)
        all_vids = sorted(os.listdir(cls_dir))
        if len(all_vids) < 3:
            continue

        train_val, test = train_test_split(all_vids, test_size=0.2, random_state=42)
        train, _ = train_test_split(train_val, test_size=0.2, random_state=42)

        train_groups = set(hmdb51_source_group(v) for v in train)
        cls_test_groups = set()
        for v in test:
            g = hmdb51_source_group(v)
            total_test += 1
            cls_test_groups.add(g)
            if g in train_groups:
                contaminated_test += 1

        for g in cls_test_groups:
            total_groups += 1
            if g in train_groups:
                contaminated_groups += 1

    return total_test, contaminated_test, total_groups, contaminated_groups


def analyze_hmdb51_official(hmdb_root, split_idx):
    """Compute group overlap for the official HMDB-51 split files."""
    splits_dir = os.path.join(hmdb_root, 'testTrainMulti_7030_splits')
    classes = sorted(
        d for d in os.listdir(hmdb_root)
        if os.path.isdir(os.path.join(hmdb_root, d))
        and d != 'testTrainMulti_7030_splits'
    )

    total_test = 0
    contaminated_test = 0
    total_groups = 0
    contaminated_groups = 0

    for cls in classes:
        split_file = os.path.join(splits_dir, f'{cls}_test_split{split_idx}.txt')
        if not os.path.exists(split_file):
            continue
        train_vids, test_vids = [], []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                vid = parts[0].replace('.avi', '')
                marker = int(parts[1])
                if marker == 1:
                    train_vids.append(vid)
                elif marker == 2:
                    test_vids.append(vid)

        train_groups = set(hmdb51_source_group(v) for v in train_vids)
        cls_test_groups = set()
        for v in test_vids:
            g = hmdb51_source_group(v)
            total_test += 1
            cls_test_groups.add(g)
            if g in train_groups:
                contaminated_test += 1

        for g in cls_test_groups:
            total_groups += 1
            if g in train_groups:
                contaminated_groups += 1

    return total_test, contaminated_test, total_groups, contaminated_groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(n, contaminated):
    rate = 100.0 * contaminated / n if n else 0
    return n, contaminated, rate


def _find_dir(data_dir, candidates):
    """Return the first existing candidate path, or the last one as a fallback."""
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.isdir(p):
            return p
    return os.path.join(data_dir, candidates[-1])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Group-level contamination analysis')
    parser.add_argument('--ucf', default=None, help='Path to UCF-101 raw data dir')
    parser.add_argument('--hmdb', default=None, help='Path to HMDB-51 raw data dir')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vnn_dir    = os.path.dirname(script_dir)
    data_root  = os.environ.get('VNN_DATA_ROOT', os.path.join(vnn_dir, 'data'))

    ucf_root  = args.ucf  or _find_dir(data_root, ['UCF-101', 'ucf101', 'UCF101', 'ucf-101'])
    hmdb_root = args.hmdb or _find_dir(data_root, ['HMDB51', 'hmdb51', 'HMDB-51', 'hmdb-51'])

    print("=" * 72)
    print("Group-Level Contamination Analysis")
    print("=" * 72)
    print()

    # ---- HMDB-51 -----------------------------------------------------------
    print("HMDB-51")
    print("-" * 50)

    if os.path.isdir(hmdb_root):
        n, c, rate = fmt(*analyze_hmdb51_buggy(hmdb_root)[:2])
        print(f"  Original (random 80/20):   {n:>5} test clips, {c:>4} contaminated ({rate:.1f}%)")

        for split_idx in [1, 2, 3]:
            result = analyze_hmdb51_official(hmdb_root, split_idx)
            if result is not None:
                n, c, rate = fmt(*result[:2])
                print(f"  Official split {split_idx}:          {n:>5} test clips, {c:>4} contaminated ({rate:.1f}%)")
    else:
        print(f"  [SKIP] HMDB-51 data not found at {hmdb_root}")

    print()

    # ---- UCF-101 -----------------------------------------------------------
    print("UCF-101")
    print("-" * 50)

    if os.path.isdir(ucf_root):
        # Buggy split: needs raw .avi files
        raw_avi_found = any(
            f.endswith('.avi')
            for cls in os.listdir(ucf_root)
            if os.path.isdir(os.path.join(ucf_root, cls))
            for f in os.listdir(os.path.join(ucf_root, cls))[:5]
        )
        if raw_avi_found:
            n, c = analyze_ucf101_buggy(ucf_root)[:2]
            rate = 100.0 * c / n if n else 0
            print(f"  Original (random 80/20):   {n:>5} test clips, {c:>4} contaminated ({rate:.1f}%)")
        else:
            print("  Original (random 80/20):   [SKIP] raw .avi files not found")

        for split_idx in [1, 2, 3]:
            result = analyze_ucf101_official(ucf_root, split_idx)
            if result is not None:
                n, c, rate = fmt(*result)
                print(f"  Official split {split_idx}:          {n:>5} test clips, {c:>4} contaminated ({rate:.1f}%)")
            else:
                print(f"  Official split {split_idx}:          [SKIP] split files not found")
    else:
        print(f"  [SKIP] UCF-101 data not found at {ucf_root}")
        print("         Pass --ucf /path/to/UCF-101 or run on the training server.")

    print()
    print("=" * 72)
    print("Definition: a test clip is 'contaminated' if its source group")
    print("(UCF-101: gXX actor ID; HMDB-51: source-video base name)")
    print("appears in at least one training clip.")
    print("=" * 72)


if __name__ == '__main__':
    main()
