"""
Download UCR/UEA time series datasets via the aeon library.

Usage:
    # Single dataset (auto-detects UCR vs UEA):
    python tools/download_ts_datasets.py --dataset ECG5000

    # Multiple datasets:
    python tools/download_ts_datasets.py --dataset ECG5000 ArticularyWordRecognition NATOPS

    # Download recommended starter pack:
    python tools/download_ts_datasets.py --starter

    # Custom output directory:
    python tools/download_ts_datasets.py --dataset ECG5000 --root ./data/ucr

Datasets are saved to:
    <root>/<DatasetName>/<DatasetName>_TRAIN.ts
    <root>/<DatasetName>/<DatasetName>_TEST.ts
"""

import argparse
import os
import sys


# Recommended datasets grouped by use case.
# Format: (name, archive, channels, length, classes, train_size, test_size)
RECOMMENDED = {
    # --- univariate (C=1) ---
    "ECG5000":         ("UCR",  1,  140,   5,   500,  4500),
    "FordA":           ("UCR",  1,  500,   2,  3601,  1320),
    "FordB":           ("UCR",  1,  500,   2,  3636,   810),
    "Wafer":           ("UCR",  1,  152,   2,  1000,  6164),
    "ElectricDevices": ("UCR",  1,   96,   7,  8926,  7711),
    # --- multivariate (C>1) ---
    "ArticularyWordRecognition": ("UEA",  9, 144, 25,  275,  300),
    "NATOPS":                    ("UEA", 24,  51,  6,  180,  180),
    "JapaneseVowels":            ("UEA", 12,  29,  9,  270,  370),
    "Epilepsy":                  ("UEA",  3, 206,  4,  137,  138),
    "BasicMotions":              ("UEA",  6, 100,  4,   40,   40),
    "CharacterTrajectories":     ("UEA",  3, 182, 20, 1422, 1436),
    "UWaveGestureLibrary":       ("UEA",  3, 315,  8,  120,  320),
    "SpokenArabicDigits":        ("UEA", 13,  93, 10, 6599, 2199),
    "Heartbeat":                 ("UEA", 61, 405,  2,  204,  205),
    "SelfRegulationSCP1":        ("UEA",  6, 896,  2,  268,  293),
    "HandMovementDirection":     ("UEA", 10, 400,  4,  160,   74),
}

STARTER_PACK = ["ECG5000", "FordA", "ArticularyWordRecognition", "NATOPS"]


def download_dataset(name: str, root: str) -> bool:
    import warnings

    try:
        # aeon >= 1.4 renamed load_classification; try the new name first.
        try:
            from aeon.datasets import load_dataset as _load
        except ImportError:
            from aeon.datasets import load_classification as _load
    except ImportError:
        print("ERROR: aeon not installed.  Run:  pip install aeon")
        sys.exit(1)

    out_dir = os.path.join(root, name)
    train_path = os.path.join(out_dir, f"{name}_TRAIN.ts")
    test_path  = os.path.join(out_dir, f"{name}_TEST.ts")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  {name}: already downloaded, skipping.")
        return True

    print(f"  {name}: downloading ...", flush=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _load(name, split="TRAIN", extract_path=root)
            _load(name, split="TEST",  extract_path=root)
        print(f"  {name}: done → {out_dir}")
        return True
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", metavar="NAME",
                        help="One or more UCR/UEA dataset names")
    parser.add_argument("--starter", action="store_true",
                        help="Download the recommended starter pack")
    parser.add_argument("--list", action="store_true",
                        help="Print recommended datasets and exit")
    parser.add_argument("--root", default=os.environ.get("UCR_ROOT", "./data/ucr"),
                        help="Download root directory (default: $UCR_ROOT or ./data/ucr)")
    args = parser.parse_args()

    if args.list:
        print("\nRecommended datasets\n" + "=" * 60)
        print(f"{'Name':<32} {'Arch':4} {'C':>3} {'T':>6} {'Cls':>4} {'Train':>6} {'Test':>6}")
        print("-" * 60)
        for name, (arch, ch, T, cls, tr, te) in RECOMMENDED.items():
            print(f"{name:<32} {arch:4} {ch:>3} {T:>6} {cls:>4} {tr:>6} {te:>6}")
        return

    targets = []
    if args.starter:
        targets = STARTER_PACK
    if args.dataset:
        targets += args.dataset

    if not targets:
        parser.print_help()
        return

    os.makedirs(args.root, exist_ok=True)
    print(f"Downloading to: {os.path.abspath(args.root)}")
    ok, fail = 0, 0
    for name in targets:
        if download_dataset(name, args.root):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
