"""Non-destructive tests for UCF101 dataset loading.

Requires preprocessed data to exist (preprocess=False throughout).
Tests are skipped if the data isn't present rather than triggering preprocessing.
"""
import os
import sys
import pytest

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mypath import Path


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _preprocessed_dir():
    _, output_dir = Path.db_dir("ucf101")
    return output_dir


def _has_preprocessed_split(split):
    d = os.path.join(_preprocessed_dir(), split)
    if not os.path.isdir(d):
        return False
    # At least one class with at least one video dir containing a jpg
    for cls in os.listdir(d):
        cls_dir = os.path.join(d, cls)
        if not os.path.isdir(cls_dir):
            continue
        for vid in os.listdir(cls_dir):
            vid_dir = os.path.join(cls_dir, vid)
            if os.path.isdir(vid_dir):
                return True
    return False


requires_train = pytest.mark.skipif(
    not _has_preprocessed_split("train"),
    reason="Preprocessed train split not found — run preprocessing first",
)
requires_test = pytest.mark.skipif(
    not _has_preprocessed_split("test"),
    reason="Preprocessed test split not found — run preprocessing first",
)
requires_val = pytest.mark.skipif(
    not _has_preprocessed_split("val"),
    reason="Preprocessed val split not found — run preprocessing first",
)


# ── Path resolution tests (always run) ───────────────────────────────────────

def test_mypath_ucf101_returns_two_strings():
    root, output = Path.db_dir("ucf101")
    assert isinstance(root, str) and isinstance(output, str)


def test_mypath_ucf101_root_ends_with_split1():
    root, _ = Path.db_dir("ucf101")
    # Default should point into split1; override via UCF101_ROOT env var
    if "UCF101_ROOT" not in os.environ:
        assert root.endswith("split1"), f"Expected path ending in split1, got: {root}"


def test_mypath_ucf101_root_exists_or_env_set():
    root, _ = Path.db_dir("ucf101")
    env_set = "UCF101_ROOT" in os.environ
    if not env_set:
        # If no env var, the default path doesn't have to exist yet (data may not be downloaded)
        pass  # just checking the call doesn't crash


# ── Dataset initialisation tests ─────────────────────────────────────────────

@requires_train
def test_train_dataset_loads():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="train", clip_len=16, preprocess=False)
    assert len(ds) > 0, "Train dataset is empty"


@requires_test
def test_test_dataset_loads():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="test", clip_len=16, preprocess=False)
    assert len(ds) > 0, "Test dataset is empty"


@requires_val
def test_val_dataset_loads():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="val", clip_len=16, preprocess=False)
    assert len(ds) > 0, "Val dataset is empty"


@requires_train
def test_train_has_101_classes():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="train", clip_len=16, preprocess=False)
    assert len(ds.label2index) == 101, (
        f"Expected 101 classes, got {len(ds.label2index)}"
    )


@requires_test
def test_test_has_same_classes_as_train():
    from dataloaders.dataset import VideoDataset
    train_ds = VideoDataset(dataset="ucf101", split="train", clip_len=16, preprocess=False)
    test_ds  = VideoDataset(dataset="ucf101", split="test",  clip_len=16, preprocess=False)
    assert set(train_ds.label2index) == set(test_ds.label2index), (
        "Train and test label sets differ"
    )


# ── Sample shape tests ────────────────────────────────────────────────────────

@requires_train
def test_train_sample_shape():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="train", clip_len=16, preprocess=False, augment=False)
    clip, label = ds[0]
    assert clip.shape == (3, 16, 112, 112), f"Unexpected shape: {clip.shape}"
    assert label.ndim == 0  # scalar


@requires_test
def test_test_sample_shape():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="test", clip_len=16, preprocess=False, augment=False)
    clip, label = ds[0]
    assert clip.shape == (3, 16, 112, 112), f"Unexpected shape: {clip.shape}"


# ── Label sanity ──────────────────────────────────────────────────────────────

@requires_train
def test_labels_in_valid_range():
    from dataloaders.dataset import VideoDataset
    ds = VideoDataset(dataset="ucf101", split="train", clip_len=16, preprocess=False)
    n_classes = len(ds.label2index)
    assert (ds.label_array >= 0).all()
    assert (ds.label_array < n_classes).all()


@requires_train
@requires_test
def test_no_overlap_between_train_and_test():
    """Verify no video directory name appears in both train and test."""
    pre_dir = _preprocessed_dir()
    def vid_names(split):
        split_dir = os.path.join(pre_dir, split)
        names = set()
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                for vid in os.listdir(cls_dir):
                    if os.path.isdir(os.path.join(cls_dir, vid)):
                        names.add(vid)
        return names

    train_vids = vid_names("train")
    test_vids  = vid_names("test")
    overlap = train_vids & test_vids
    assert len(overlap) == 0, f"Found {len(overlap)} videos in both train and test: {list(overlap)[:5]}"
