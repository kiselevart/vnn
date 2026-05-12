import os
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from mypath import Path
from utils.video_utils import calculate_video_flow


def _video_preprocess_job(args):
    """Module-level worker for parallel video preprocessing.

    Must be at module level (not a method) so multiprocessing can pickle it.
    args: (src_path, target_dir, clip_len, resize_height, resize_width)
    """
    src_path, target_dir, clip_len, resize_height, resize_width = args

    if os.path.isdir(target_dir) and any(f.endswith('.jpg') for f in os.listdir(target_dir)):
        return

    capture = cv2.VideoCapture(src_path)
    if not capture.isOpened():
        capture.release()
        return

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count < clip_len:
        capture.release()
        return

    freq = 4
    while freq > 1 and frame_count // freq < clip_len:
        freq -= 1

    os.makedirs(target_dir, exist_ok=True)
    count = i = 0
    retaining = True
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is None:
            continue
        if count % freq == 0:
            if frame_height != resize_height or frame_width != resize_width:
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(os.path.join(target_dir, '0000{}.jpg'.format(i)), frame)
            i += 1
        count += 1
    capture.release()

    if i == 0:
        try:
            os.rmdir(target_dir)
        except OSError:
            pass
        return

    frame_paths = sorted(
        os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.jpg')
    )
    imgs = [cv2.imread(p) for p in frame_paths]
    imgs = [img for img in imgs if img is not None]
    if len(imgs) < 2:
        return
    video_tensor = torch.from_numpy(np.stack(imgs, 0)).permute(3, 0, 1, 2).float()
    flow = calculate_video_flow(video_tensor)
    np.save(os.path.join(target_dir, 'flow.npy'), flow.numpy())


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False, augment=True, ucf_split=1):
        self.root_dir, base_output_dir = Path.db_dir(dataset)
        # UCF101 and HMDB51 have 3 official splits; store each split's frames separately
        # so multiple splits can coexist on disk without re-extraction.
        if dataset.lower() in ('ucf101', 'hmdb51'):
            self.output_dir = os.path.join(base_output_dir, f'split{ucf_split}')
        else:
            self.output_dir = base_output_dir
        self.ucf_split = ucf_split
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.augment = augment

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Per-channel BGR mean pixel values (C3D/Sports-1M convention).
        # Stored as an attribute so consumers (e.g. FlowDatasetWrapper) can
        # denormalize without hardcoding these values themselves.
        self.mean = np.array([90.0, 98.0, 102.0], dtype=np.float32)  # [B, G, R]

        # Detect whether root_dir has pre-split structure (train/test subdirs)
        # or flat class folders that need splitting
        self.pre_split = (
            os.path.isdir(os.path.join(self.root_dir, 'train')) and
            os.path.isdir(os.path.join(self.root_dir, 'test'))
        )

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.'
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print(f'==> Preprocessing {dataset} dataset — extracting frames from videos.')
            print(f'    Source : {self.root_dir}')
            print(f'    Output : {self.output_dir}')
            print(f'    This runs once and may take 10–60+ minutes depending on dataset size.')
            self.preprocess()
            self._invalidate_filelist_cache(split)

        cache_path = os.path.join(self.output_dir, f'filelist_{split}.pkl')
        if os.path.exists(cache_path):
            import pickle
            with open(cache_path, 'rb') as f:
                self.fnames, labels = pickle.load(f)
            print(f'Number of {split} videos: {len(self.fnames):d} (from cache)')
        else:
            import pickle
            self.fnames, labels = [], []
            skipped = 0
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, label)):
                    fpath = os.path.join(folder, label, fname)
                    if not os.path.isdir(fpath):
                        continue
                    n_frames = sum(1 for f in os.listdir(fpath) if f.endswith('.jpg'))
                    if n_frames < 2:
                        skipped += 1
                        continue
                    self.fnames.append(fpath)
                    labels.append(label)
            if skipped:
                print(f'  [INFO] Skipped {skipped} videos with fewer than 2 frames.')
            print(f'Number of {split} videos: {len(self.fnames):d}')
            with open(cache_path, 'wb') as f:
                pickle.dump((self.fnames, labels), f)

        assert len(labels) == len(self.fnames)

        # Prepare a mapping between the label names (strings) and indices (ints).
        # Use ALL class directories (same list as the outer loop), not just those
        # that happen to have valid videos in this split. If a class loses all its
        # videos to the short-frame filter, sorted(set(labels)) would omit it and
        # shift every subsequent class index — corrupting val/test label alignment
        # with the train-derived model outputs.
        self.label2index = {label: index for index, label in enumerate(sorted(os.listdir(folder)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        if self.augment:
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
        else:
            buffer = self.center_crop(buffer, self.clip_len, self.crop_size)
        buffer = self.ensure_clip_len(buffer, self.clip_len)
        labels = np.array(self.label_array[index])

        if self.augment:
            buffer = self.randomflip(buffer)
            buffer = self.color_jitter(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def _invalidate_filelist_cache(self, split):
        cache_path = os.path.join(self.output_dir, f'filelist_{split}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        """Check if preprocessed frames exist with correct dimensions."""
        if not os.path.exists(self.output_dir):
            return False
        train_dir = os.path.join(self.output_dir, 'train')
        if not os.path.exists(train_dir):
            return False

        for ii, video_class in enumerate(os.listdir(train_dir)):
            class_dir = os.path.join(train_dir, video_class)
            if not os.path.isdir(class_dir):
                continue
            for video in os.listdir(class_dir):
                video_dir = os.path.join(class_dir, video)
                if not os.path.isdir(video_dir):
                    continue
                frames = sorted(os.listdir(video_dir))
                if not frames:
                    return False
                image = cv2.imread(os.path.join(video_dir, frames[0]))
                if image is None or image.shape[0] != 128 or image.shape[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)

        train_list = os.path.join(self.root_dir, 'ucfTrainTestlist', f'trainlist0{self.ucf_split}.txt')
        test_list  = os.path.join(self.root_dir, 'ucfTrainTestlist', f'testlist0{self.ucf_split}.txt')
        hmdb_splits_dir = os.path.join(self.root_dir, 'testTrainMulti_7030_splits')

        if self._detect_ssv2():
            self._preprocess_ssv2()
        elif os.path.exists(train_list) and os.path.exists(test_list):
            # UCF101 official splits — group-aware, no performer leakage
            self._preprocess_official_splits(train_list, test_list)
        elif os.path.isdir(hmdb_splits_dir):
            # HMDB51 official split files alongside raw videos
            self._preprocess_hmdb51_splits(hmdb_splits_dir)
        elif self.pre_split:
            # Data already split into train/val/test with class subfolders of videos
            self._preprocess_pre_split()
        elif self._is_preextracted_frames():
            # Pre-extracted JPEG frames (e.g. Kaggle HMDB51): resize and randomly split
            self._preprocess_from_frames()
        else:
            # Flat class folders of raw videos — random split
            self._preprocess_flat()

        print('Preprocessing finished.')

    def _is_video_file(self, path):
        return os.path.isfile(path) and path.lower().endswith(('.avi', '.mp4', '.mkv', '.mpg', '.mpeg', '.mov', '.webm'))

    def _collect_video_entries(self, class_dir):
        """Collect supported videos in a class directory.

        Supports:
          1) Direct video files under class_dir
          2) One-level nested folders containing video files (e.g., UCF11 groups)

        Returns paths relative to class_dir suitable for joining later.
        """
        entries = []
        for name in sorted(os.listdir(class_dir)):
            full_path = os.path.join(class_dir, name)
            if self._is_video_file(full_path):
                entries.append(name)
                continue

            if os.path.isdir(full_path):
                for nested in sorted(os.listdir(full_path)):
                    nested_path = os.path.join(full_path, nested)
                    if self._is_video_file(nested_path):
                        entries.append(os.path.join(name, nested))

        return entries

    def _preprocess_official_splits(self, train_list_path, test_list_path):
        """Preprocess using official UCF101 split files (trainlist01.txt / testlist01.txt).

        Val is carved from the official training set by splitting performer *groups*
        (the gXX part of the filename) — all clips of a group go to the same partition,
        preventing same-performer leakage between train and val.
        """
        def parse_list(path):
            entries = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('/')
                    if len(parts) != 2:
                        continue
                    cls = parts[0]
                    vid = parts[1].split()[0]  # strip label if present
                    entries.append((cls, vid))
            return entries

        train_entries = parse_list(train_list_path)
        test_entries  = parse_list(test_list_path)

        # Group train entries by (class, performer group)
        class_groups = defaultdict(lambda: defaultdict(list))  # cls -> gXX -> [vids]
        for cls, vid in train_entries:
            m = re.match(r'v_\w+_(g\d+)_c\d+\.avi', vid)
            group = m.group(1) if m else 'g00'
            class_groups[cls][group].append(vid)

        # Split groups 80/20 into train/val per class, seeded for reproducibility
        rng = np.random.RandomState(42)
        final_train, final_val = [], []
        for cls in sorted(class_groups.keys()):
            groups = sorted(class_groups[cls].keys())
            rng.shuffle(groups)
            n_val = max(1, int(len(groups) * 0.2))
            val_groups = set(groups[:n_val])
            for group in groups:
                for vid in class_groups[cls][group]:
                    (final_val if group in val_groups else final_train).append((cls, vid))

        for split_name, entries in [('train', final_train), ('val', final_val), ('test', test_entries)]:
            print(f'  {split_name}: {len(entries)} videos')
            for cls, vid in tqdm(entries, desc=f'Processing {split_name}'):
                save_dir = os.path.join(self.output_dir, split_name, cls)
                os.makedirs(save_dir, exist_ok=True)
                self.process_video(vid, cls, save_dir)

    def _carve_val_from_train(self, train_entries, val_ratio=0.2, seed=42):
        """Group-aware val split: all clips from the same _gXX_ performer group
        stay together so no performer leaks between train and val.
        Returns (final_train, val) as lists of (class, video) tuples.
        """
        rng = np.random.RandomState(seed)
        class_groups = defaultdict(lambda: defaultdict(list))
        for cls, vid in train_entries:
            m = re.match(r'v_\w+_(g\d+)_c\d+\.avi', vid)
            group = m.group(1) if m else 'g00'
            class_groups[cls][group].append(vid)

        final_train, final_val = [], []
        for cls in sorted(class_groups):
            groups = sorted(class_groups[cls])
            rng.shuffle(groups)
            n_val = max(1, int(len(groups) * val_ratio))
            val_groups = set(groups[:n_val])
            for g in groups:
                for vid in class_groups[cls][g]:
                    (final_val if g in val_groups else final_train).append((cls, vid))
        return final_train, final_val

    def _preprocess_pre_split(self):
        """Preprocess when root_dir has train/test (and optionally val) class subfolders.

        If no val/ dir exists, carves val from train using performer-group isolation.
        """
        has_val = os.path.isdir(os.path.join(self.root_dir, 'val'))

        if has_val:
            # All three dirs present — just extract frames for each
            for split in ['train', 'val', 'test']:
                split_src = os.path.join(self.root_dir, split)
                if not os.path.isdir(split_src):
                    continue
                for action_name in sorted(os.listdir(split_src)):
                    action_dir = os.path.join(split_src, action_name)
                    if not os.path.isdir(action_dir):
                        continue
                    save_dir = os.path.join(self.output_dir, split, action_name)
                    os.makedirs(save_dir, exist_ok=True)
                    for video in self._collect_video_entries(action_dir):
                        self.process_video(video, os.path.join(split, action_name), save_dir)
        else:
            # No val dir — collect train entries, carve val, then process all three
            train_src = os.path.join(self.root_dir, 'train')
            train_entries = []
            for action_name in sorted(os.listdir(train_src)):
                action_dir = os.path.join(train_src, action_name)
                if not os.path.isdir(action_dir):
                    continue
                for video in self._collect_video_entries(action_dir):
                    train_entries.append((action_name, video))

            final_train, final_val = self._carve_val_from_train(train_entries)
            print(f'  train: {len(final_train)}  val (carved): {len(final_val)}')

            for split_name, entries in [('train', final_train), ('val', final_val)]:
                for cls, vid in tqdm(entries, desc=f'Processing {split_name}'):
                    save_dir = os.path.join(self.output_dir, split_name, cls)
                    os.makedirs(save_dir, exist_ok=True)
                    self.process_video(vid, os.path.join('train', cls), save_dir)

            # Process test as-is
            test_src = os.path.join(self.root_dir, 'test')
            for action_name in sorted(os.listdir(test_src)):
                action_dir = os.path.join(test_src, action_name)
                if not os.path.isdir(action_dir):
                    continue
                save_dir = os.path.join(self.output_dir, 'test', action_name)
                os.makedirs(save_dir, exist_ok=True)
                for video in tqdm(self._collect_video_entries(action_dir), desc=f'test/{action_name}'):
                    self.process_video(video, os.path.join('test', action_name), save_dir)

    def _preprocess_flat(self):
        """Preprocess when root_dir has flat class folders (original UCF101 layout)."""
        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            if os.path.isdir(file_path):
                video_files = self._collect_video_entries(file_path)
                if len(video_files) < 3:
                    continue

                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                train_dir = os.path.join(self.output_dir, 'train', file)
                val_dir = os.path.join(self.output_dir, 'val', file)
                test_dir = os.path.join(self.output_dir, 'test', file)

                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

                for video in train:
                    self.process_video(video, file, train_dir)

                for video in val:
                    self.process_video(video, file, val_dir)

                for video in test:
                    self.process_video(video, file, test_dir)

    def _is_preextracted_frames(self):
        """Return True if root_dir holds class→video_dir→jpg frames (no raw video files at class level)."""
        for cls in sorted(os.listdir(self.root_dir)):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for item in sorted(os.listdir(cls_dir)):
                item_path = os.path.join(cls_dir, item)
                if os.path.isdir(item_path):
                    if any(f.endswith('.jpg') for f in os.listdir(item_path)):
                        return True
                elif self._is_video_file(item_path):
                    return False
            break  # Only probe the first class
        return False

    def _resize_frames_to_dir(self, src_dir, dst_dir):
        """Resize all jpg frames from src_dir to resize_height×resize_width and write to dst_dir."""
        frames = sorted(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        if not frames:
            return
        os.makedirs(dst_dir, exist_ok=True)
        for i, fname in enumerate(frames):
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                continue
            if img.shape[0] != self.resize_height or img.shape[1] != self.resize_width:
                img = cv2.resize(img, (self.resize_width, self.resize_height))
            cv2.imwrite(os.path.join(dst_dir, f'{i:05d}.jpg'), img)

    def _preprocess_from_frames(self):
        """Preprocess a pre-extracted frame dataset (class/video_dir/jpg).

        Resizes frames to resize_height×resize_width, splits 70/10/20 into
        train/val/test using a seed derived from ucf_split so different split
        numbers yield different random partitions.
        """
        # Seed varies per ucf_split so --avg_splits gives 3 independent partitions.
        rng = np.random.RandomState(42 + self.ucf_split)

        for cls in tqdm(sorted(os.listdir(self.root_dir)), desc="Preprocessing HMDB51 classes"):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            video_dirs = sorted(
                d for d in os.listdir(cls_dir)
                if os.path.isdir(os.path.join(cls_dir, d))
            )
            if not video_dirs:
                continue

            video_dirs = list(rng.permutation(video_dirs))
            n = len(video_dirs)
            n_test = max(1, int(n * 0.2))
            n_val  = max(1, int(n * 0.1))
            splits_map = [
                ('test',  video_dirs[:n_test]),
                ('val',   video_dirs[n_test:n_test + n_val]),
                ('train', video_dirs[n_test + n_val:]),
            ]

            for split_name, dirs in splits_map:
                split_cls_dir = os.path.join(self.output_dir, split_name, cls)
                os.makedirs(split_cls_dir, exist_ok=True)
                for vid_dir in dirs:
                    src = os.path.join(cls_dir, vid_dir)
                    dst = os.path.join(split_cls_dir, vid_dir)
                    if os.path.exists(dst):
                        continue
                    self._resize_frames_to_dir(src, dst)
                    self._compute_and_save_flow(dst)

    def _preprocess_hmdb51_splits(self, splits_dir):
        """Preprocess HMDB51 using official per-class split files.

        Split file format (one file per class per split):
          testTrainMulti_7030_splits/<class>_test_split{N}.txt
          Each line: 'video.avi 1' (train), 'video.avi 2' (test), 'video.avi 0' (unused)
        """
        train_entries, test_entries = [], []

        for cls in sorted(os.listdir(self.root_dir)):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            split_file = os.path.join(splits_dir, f'{cls}_test_split{self.ucf_split}.txt')
            if not os.path.exists(split_file):
                continue
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    vid, marker = parts[0], int(parts[1])
                    if marker == 1:
                        train_entries.append((cls, vid))
                    elif marker == 2:
                        test_entries.append((cls, vid))

        # Stratified val carve from train (per class, 15%)
        rng = np.random.RandomState(42)
        class_trains = defaultdict(list)
        for cls, vid in train_entries:
            class_trains[cls].append(vid)
        final_train, final_val = [], []
        for cls in sorted(class_trains):
            vids = class_trains[cls]
            rng.shuffle(vids)
            n_val = max(1, int(len(vids) * 0.15))
            final_val.extend((cls, v) for v in vids[:n_val])
            final_train.extend((cls, v) for v in vids[n_val:])

        for split_name, entries in [('train', final_train), ('val', final_val), ('test', test_entries)]:
            print(f'  {split_name}: {len(entries)} videos')
            for cls, vid in tqdm(entries, desc=f'Processing {split_name}'):
                save_dir = os.path.join(self.output_dir, split_name, cls)
                os.makedirs(save_dir, exist_ok=True)
                vid_stem = os.path.splitext(vid)[0]
                src_frames = os.path.join(self.root_dir, cls, vid_stem)
                if os.path.isdir(src_frames):
                    # Pre-extracted frames layout: root/class/video_stem/frame.jpg
                    dst = os.path.join(save_dir, vid_stem)
                    if not os.path.exists(dst):
                        self._resize_frames_to_dir(src_frames, dst)
                        self._compute_and_save_flow(dst)
                else:
                    self.process_video(vid, cls, save_dir)

    def _detect_ssv2(self):
        return os.path.exists(os.path.join(self.root_dir, 'labels', 'labels.json'))

    def _preprocess_ssv2(self):
        import json, csv
        from multiprocessing import Pool

        with open(os.path.join(self.root_dir, 'labels', 'labels.json')) as f:
            json.load(f)  # validate; actual class dirs come from template strings

        def process_split(entries, split_name):
            jobs = []
            for vid_id, template in entries:
                save_dir = os.path.join(self.output_dir, split_name, template)
                os.makedirs(save_dir, exist_ok=True)
                for ext in ('.webm', '.mp4'):
                    p = os.path.join(self.root_dir, vid_id + ext)
                    if os.path.exists(p):
                        jobs.append((p, os.path.join(save_dir, vid_id),
                                     self.clip_len, self.resize_height, self.resize_width))
                        break

            n_workers = min(os.cpu_count() or 4, 32)
            print(f'  {split_name}: {len(jobs)} videos, {n_workers} workers')
            with Pool(n_workers) as pool:
                for _ in tqdm(pool.imap_unordered(_video_preprocess_job, jobs, chunksize=8),
                              total=len(jobs), desc=f'Processing {split_name}'):
                    pass

        with open(os.path.join(self.root_dir, 'labels', 'train.json')) as f:
            train_ann = json.load(f)
        train_entries = [(e['id'], e['template']) for e in train_ann]
        print(f'  train: {len(train_entries)} videos')
        process_split(train_entries, 'train')

        with open(os.path.join(self.root_dir, 'labels', 'validation.json')) as f:
            val_ann = json.load(f)
        val_entries = [(e['id'], e['template']) for e in val_ann]
        print(f'  val: {len(val_entries)} videos')
        process_split(val_entries, 'val')

        test_entries = []
        with open(os.path.join(self.root_dir, 'labels', 'test-answers.csv')) as f:
            for row in csv.reader(f, delimiter=';'):
                if len(row) >= 2:
                    test_entries.append((row[0], row[1]))
        print(f'  test: {len(test_entries)} videos')
        process_split(test_entries, 'test')

    def _extract_video_to_dir(self, src_path, target_dir):
        _video_preprocess_job((src_path, target_dir, self.clip_len, self.resize_height, self.resize_width))

    def process_video(self, video, action_name, save_dir):
        src_path = os.path.join(self.root_dir, action_name, video)
        video_filename = os.path.splitext(os.path.basename(video))[0]
        target_dir = os.path.join(save_dir, video_filename)
        self._extract_video_to_dir(src_path, target_dir)

    def _compute_and_save_flow(self, video_dir):
        """Compute optical flow from raw frames in video_dir and save to flow.npy.

        Operates on full-resolution frames (before any crop/normalize) so the
        stored flow can be spatially cropped at load time to match any RGB crop.
        Output: float32 array [2, T, H, W] saved as flow.npy alongside the frames.
        """
        frame_paths = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir) if f.endswith('.jpg')
        ])
        if len(frame_paths) < 2:
            return

        imgs = [cv2.imread(p) for p in frame_paths]
        imgs = [img for img in imgs if img is not None]
        if len(imgs) < 2:
            return

        # [T, H, W, C] → tensor [C, T, H, W], raw uint8 pixel values (no normalization)
        video_tensor = torch.from_numpy(
            np.stack(imgs, axis=0)
        ).permute(3, 0, 1, 2).float()

        flow = calculate_video_flow(video_tensor)  # [2, T, H, W]
        np.save(os.path.join(video_dir, "flow.npy"), flow.numpy())

    def ensure_flows(self):
        """Compute and cache flow.npy for any video directory that lacks one.

        Idempotent — skips videos that already have flow.npy. Intended for
        datasets preprocessed before flow caching was introduced.
        """
        missing = [
            d for d in self.fnames
            if not os.path.exists(os.path.join(d, "flow.npy"))
        ]
        if not missing:
            return
        print(f"==> Computing optical flow for {len(missing)} videos (one-time)...")
        for video_dir in tqdm(missing, desc="Flow"):
            self._compute_and_save_flow(video_dir)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i in range(len(buffer)):
                buffer[i] = cv2.flip(buffer[i], flipCode=1)

        return buffer

    def color_jitter(self, buffer, brightness=0.3, contrast=0.3):
        """Random brightness and contrast jitter applied uniformly across all frames.

        buffer: [T, H, W, C] float32 with raw pixel values in [0, 255].
        Returns buffer in the same range and dtype.
        """
        alpha = 1.0 + np.random.uniform(-contrast, contrast)   # contrast scale
        beta = np.random.uniform(-brightness, brightness) * 255.0  # brightness shift
        return np.clip(alpha * buffer + beta, 0.0, 255.0).astype(buffer.dtype)


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= self.mean
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir) if img.endswith('.jpg')])
        frame_count = len(frames)
        if frame_count == 0:
            return np.empty((0, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            if frame is None:
                frame = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.float32)
            else:
                frame = np.array(frame).astype(np.float64)
            buffer[i] = frame

        return buffer

    def ensure_clip_len(self, buffer, clip_len):
        """Guarantee temporal length equals clip_len for batching safety."""
        num_frames = buffer.shape[0]
        if num_frames == clip_len:
            return buffer

        if num_frames == 0:
            return np.zeros((clip_len, self.crop_size, self.crop_size, 3), dtype=np.float32)

        if num_frames > clip_len:
            return buffer[:clip_len]

        pad_count = clip_len - num_frames
        pad_frames = np.repeat(buffer[-1:, :, :, :], repeats=pad_count, axis=0)
        return np.concatenate([buffer, pad_frames], axis=0)

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        max_time = buffer.shape[0] - clip_len
        time_index = np.random.randint(max_time + 1) if max_time > 0 else 0

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(max(1, buffer.shape[1] - crop_size))
        width_index = np.random.randint(max(1, buffer.shape[2] - crop_size))

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def center_crop(self, buffer, clip_len, crop_size):
        # Center temporal clip
        time_index = max(0, (buffer.shape[0] - clip_len) // 2)
        height_index = max(0, (buffer.shape[1] - crop_size) // 2)
        width_index = max(0, (buffer.shape[2] - crop_size) // 2)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
