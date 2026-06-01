import os
import re
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

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

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False, augment=True, ucf_split=1,
                 num_clips=1, num_crops=1):
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
        self.num_clips = num_clips
        self.num_crops = num_crops

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Per-channel BGR mean pixel values (C3D/Sports-1M convention).
        self.mean = np.array([90.0, 98.0, 102.0], dtype=np.float32)  # [B, G, R]

        # Detect whether root_dir has pre-split structure (split1/split2/split3 layout)
        self.pre_split = (
            os.path.isdir(os.path.join(self.root_dir, 'split1')) or
            os.path.isdir(os.path.join(self.root_dir, 'train'))
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
            for s in ['train', 'val', 'test']:
                self._invalidate_filelist_cache(s)

        self.fnames, labels = [], []
        cache_path = os.path.join(self.output_dir, f'filelist_{split}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.fnames, labels = pickle.load(f)
            if self.fnames:
                print(f'Number of {split} videos: {len(self.fnames):d} (from cache)')
            else:
                os.remove(cache_path)
        if not os.path.exists(cache_path):
            self.fnames, labels = [], []
            skipped = 0
            if os.path.exists(folder):
                for label in sorted(os.listdir(folder)):
                    label_dir = os.path.join(folder, label)
                    if not os.path.isdir(label_dir):
                        continue
                    for fname in os.listdir(label_dir):
                        fpath = os.path.join(label_dir, fname)
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

        self.label2index = {label: index for index, label in enumerate(sorted(os.listdir(folder)))} if os.path.exists(folder) else {}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101" and self.label2index:
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                os.makedirs('dataloaders', exist_ok=True)
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51' and self.label2index:
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                os.makedirs('dataloaders', exist_ok=True)
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def _get_view_indices(self, T, H, W):
        crop = self.crop_size
        h_c = max(0, (H - crop) // 2)
        w_c = max(0, (W - crop) // 2)

        if self.num_crops == 1:
            spatial = [(h_c, w_c)]
        else:
            if W >= H:
                spatial = [(h_c, 0), (h_c, w_c), (h_c, max(0, W - crop))]
            else:
                spatial = [(0, w_c), (h_c, w_c), (max(0, H - crop), w_c)]

        indices = []
        max_t = max(0, T - self.clip_len)
        for ci in range(self.num_clips):
            if self.num_clips == 1:
                t = max_t // 2
            else:
                t = int(round(max_t * ci / (self.num_clips - 1)))
                t = max(0, min(t, max_t))
            for h, w in spatial:
                indices.append((t, h, w))
        return indices

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])

        if self.augment:
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.ensure_clip_len(buffer, self.clip_len)
            buffer = self.randomflip(buffer)
            buffer = self.color_jitter(buffer)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            return torch.from_numpy(buffer), torch.from_numpy(labels)

        T, H, W = buffer.shape[0], buffer.shape[1], buffer.shape[2]
        views = []
        for t, h, w in self._get_view_indices(T, H, W):
            clip = buffer[t:t + self.clip_len, h:h + self.crop_size, w:w + self.crop_size].copy()
            clip = self.ensure_clip_len(clip, self.clip_len)
            clip = self.normalize(clip)
            clip = self.to_tensor(clip)
            views.append(torch.from_numpy(clip))

        out = torch.stack(views, 0) if len(views) > 1 else views[0]
        return out, torch.from_numpy(labels)

    def _invalidate_filelist_cache(self, split):
        cache_path = os.path.join(self.output_dir, f'filelist_{split}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def check_integrity(self):
        return os.path.exists(self.root_dir)

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        train_dir = os.path.join(self.output_dir, 'train')
        if not os.path.exists(train_dir):
            return False

        found_any = False
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
                if image is None or image.shape[0] != self.resize_height or image.shape[1] != self.resize_width:
                    return False
                found_any = True
                break
            if found_any or ii == 10:
                break
        return found_any

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Process standard directory structures or switch variants
        if os.path.isdir(os.path.join(self.root_dir, 'split1')):
            for split_idx in [1, 2, 3]:
                split_name = f'split{split_idx}'
                src_split_dir = os.path.join(self.root_dir, split_name)
                if not os.path.isdir(src_split_dir):
                    continue
                print(f"==> Preprocessing structured execution directory: {split_name}")
                self._preprocess_nested_ucf_split(src_split_dir, split_name)
        else:
            train_list = os.path.join(self.root_dir, 'ucfTrainTestlist', f'trainlist0{self.ucf_split}.txt')
            test_list  = os.path.join(self.root_dir, 'ucfTrainTestlist', f'testlist0{self.ucf_split}.txt')
            hmdb_splits_dir = os.path.join(self.root_dir, 'testTrainMulti_7030_splits')

            if self._detect_diving48():
                self._preprocess_diving48()
            elif self._detect_ssv2():
                self._preprocess_ssv2()
            elif os.path.exists(train_list) and os.path.exists(test_list):
                self._preprocess_official_splits(train_list, test_list)
            elif os.path.isdir(hmdb_splits_dir):
                self._preprocess_hmdb51_splits(hmdb_splits_dir)
            elif self._is_preextracted_frames():
                self._preprocess_from_frames()
            else:
                self._preprocess_flat()

        print('Preprocessing finished.')

    def _preprocess_nested_ucf_split(self, src_split_dir, split_name):
        """Processes an explicit layout file map, slicing 15% of train for val

        safeguarding group context from leakage via multiprocessing execution.
        """
        rng = np.random.RandomState(42)
        jobs = []

        train_src = os.path.join(src_split_dir, 'train')
        if os.path.isdir(train_src):
            for action_name in sorted(os.listdir(train_src)):
                action_dir = os.path.join(train_src, action_name)
                if not os.path.isdir(action_dir):
                    continue

                videos = self._collect_video_entries(action_dir)
                group_to_vids = defaultdict(list)
                for vid in videos:
                    m = re.match(r'v_\w+_(g\d+)_c\d+\.avi', os.path.basename(vid))
                    group = m.group(1) if m else 'g00'
                    group_to_vids[group].append(vid)

                distinct_groups = sorted(list(group_to_vids.keys()))
                rng.shuffle(distinct_groups)
                n_val = max(1, int(len(distinct_groups) * 0.15))
                val_groups = set(distinct_groups[:n_val])

                for group in distinct_groups:
                    target_partition = 'val' if group in val_groups else 'train'
                    base_dir = os.path.dirname(self.output_dir)
                    save_dir = os.path.join(base_dir, split_name, target_partition, action_name)
                    os.makedirs(save_dir, exist_ok=True)

                    for vid in group_to_vids[group]:
                        src_path = os.path.join(action_dir, vid)
                        video_filename = os.path.splitext(os.path.basename(vid))[0]
                        target_dir = os.path.join(save_dir, video_filename)
                        jobs.append((src_path, target_dir, self.clip_len, self.resize_height, self.resize_width))

        test_src = os.path.join(src_split_dir, 'test')
        if os.path.isdir(test_src):
            for action_name in sorted(os.listdir(test_src)):
                action_dir = os.path.join(test_src, action_name)
                if not os.path.isdir(action_dir):
                    continue

                save_dir = os.path.join(os.path.dirname(self.output_dir), split_name, 'test', action_name)
                os.makedirs(save_dir, exist_ok=True)

                for vid in self._collect_video_entries(action_dir):
                    src_path = os.path.join(action_dir, vid)
                    video_filename = os.path.splitext(os.path.basename(vid))[0]
                    target_dir = os.path.join(save_dir, video_filename)
                    jobs.append((src_path, target_dir, self.clip_len, self.resize_height, self.resize_width))

        if jobs:
            n_workers = min(os.cpu_count() or 4, 32)
            print(f"   Executing {len(jobs)} parallelized worker jobs via pool allocation...")
            with Pool(n_workers) as pool:
                for _ in tqdm(pool.imap_unordered(_video_preprocess_job, jobs, chunksize=8),
                              total=len(jobs), desc=f"Extracting {split_name}"):
                    pass

    def _is_video_file(self, path):
        return os.path.isfile(path) and path.lower().endswith(('.avi', '.mp4', '.mkv', '.mpg', '.mpeg', '.mov', '.webm'))

    def _collect_video_entries(self, class_dir):
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
        def parse_list(path):
            entries = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    file_path = tokens[0]
                    if '/' in file_path:
                        cls, vid = file_path.split('/')
                    else:
                        vid = file_path
                        match_cls = re.match(r'v_(\w+)_g\d+_c\d+', vid)
                        cls = match_cls.group(1) if match_cls else "Unknown"
                    entries.append((cls, vid))
            return entries

        train_entries = parse_list(train_list_path)
        test_entries  = parse_list(test_list_path)

        class_groups = defaultdict(lambda: defaultdict(list))
        for cls, vid in train_entries:
            m = re.match(r'v_\w+_(g\d+)_c\d+\.avi', vid)
            group = m.group(1) if m else 'g00'
            class_groups[cls][group].append(vid)

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
            jobs = []
            for cls, vid in entries:
                save_dir = os.path.join(self.output_dir, split_name, cls)
                os.makedirs(save_dir, exist_ok=True)
                src_path = os.path.join(self.root_dir, cls, vid)
                video_filename = os.path.splitext(os.path.basename(vid))[0]
                target_dir = os.path.join(save_dir, video_filename)
                jobs.append((src_path, target_dir, self.clip_len, self.resize_height, self.resize_width))

            n_workers = min(os.cpu_count() or 4, 32)
            with Pool(n_workers) as pool:
                for _ in tqdm(pool.imap_unordered(_video_preprocess_job, jobs, chunksize=8),
                              total=len(jobs), desc=f'Processing {split_name}'):
                    pass

    def _preprocess_flat(self):
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            if os.path.isdir(file_path):
                video_files = self._collect_video_entries(file_path)
                if len(video_files) < 3:
                    continue

                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                for split_name, split_set in [('train', train), ('val', val), ('test', test)]:
                    target_dir = os.path.join(self.output_dir, split_name, file)
                    os.makedirs(target_dir, exist_ok=True)
                    for video in split_set:
                        self.process_video(video, file, target_dir)

    def _is_preextracted_frames(self):
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
            break
        return False

    def _resize_frames_to_dir(self, src_dir, dst_dir):
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
        rng = np.random.RandomState(42 + self.ucf_split)
        for cls in tqdm(sorted(os.listdir(self.root_dir)), desc="Preprocessing HMDB51 classes"):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            video_dirs = sorted(d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d)))
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
                    dst = os.path.join(save_dir, vid_stem)
                    if not os.path.exists(dst):
                        self._resize_frames_to_dir(src_frames, dst)
                        self._compute_and_save_flow(dst)
                else:
                    self.process_video(vid, cls, save_dir)

    def _detect_diving48(self):
        return (
            os.path.exists(os.path.join(self.root_dir, 'diving48_v2_train.json')) and
            os.path.exists(os.path.join(self.root_dir, 'diving48_v2_test.json'))
        )

    def _find_diving48_rgb_dir(self, vid_name):
        """Return the source RGB frame directory for a Diving48 clip, or None."""
        # Common layouts after extracting the official archives:
        #   <root>/rgb/<vid_name>/      (rgb archive extracted as rgb/)
        #   <root>/frames/<vid_name>/   (alternative name)
        #   <root>/<vid_name>/          (flat layout)
        for sub in ('rgb', 'frames', ''):
            candidate = os.path.join(self.root_dir, sub, vid_name) if sub else \
                        os.path.join(self.root_dir, vid_name)
            if os.path.isdir(candidate) and any(f.endswith('.jpg') for f in os.listdir(candidate)):
                return candidate
        return None

    def _preprocess_diving48(self):
        """Preprocess Diving48 using pre-extracted RGB frames.

        Ignores the dataset's TVL1 optical flow to keep flow statistics
        identical to UCF101/HMDB51 (both use Farneback with 0.05 scale).
        Flow is recomputed from the resized RGB frames using calculate_video_flow.

        Expected directory layout after extracting the official archives:
            data/diving48/
              diving48_v2_train.json
              diving48_v2_test.json
              rgb/<vid_name>/<frame>.jpg   (pre-extracted RGB frames)
        """
        import json
        rng = np.random.RandomState(42)

        with open(os.path.join(self.root_dir, 'diving48_v2_train.json')) as f:
            train_ann = json.load(f)
        with open(os.path.join(self.root_dir, 'diving48_v2_test.json')) as f:
            test_ann = json.load(f)

        # Carve 15% of each class for val (no actor groups in diving)
        label_to_entries = defaultdict(list)
        for entry in train_ann:
            label_to_entries[entry['label']].append(entry)

        final_train, final_val = [], []
        for label_id in sorted(label_to_entries):
            entries = list(label_to_entries[label_id])
            rng.shuffle(entries)
            n_val = max(1, int(len(entries) * 0.15))
            final_val.extend(entries[:n_val])
            final_train.extend(entries[n_val:])

        missing = 0
        for split_name, entries in [('train', final_train), ('val', final_val), ('test', list(test_ann))]:
            print(f'  {split_name}: {len(entries)} clips')
            for entry in tqdm(entries, desc=f'Processing {split_name}'):
                # Zero-padded integer label → sorted() order matches integer class index
                label_folder = f"{entry['label']:02d}"
                vid_name = entry['vid_name']

                src_dir = self._find_diving48_rgb_dir(vid_name)
                if src_dir is None:
                    missing += 1
                    continue

                dst_dir = os.path.join(self.output_dir, split_name, label_folder, vid_name)
                if os.path.exists(dst_dir) and any(f.endswith('.jpg') for f in os.listdir(dst_dir)):
                    # Already processed; ensure flow exists
                    if not os.path.exists(os.path.join(dst_dir, 'flow.npy')):
                        self._compute_and_save_flow(dst_dir)
                    continue

                self._resize_frames_to_dir(src_dir, dst_dir)
                self._compute_and_save_flow(dst_dir)

        if missing:
            print(f'  [WARN] {missing} clips had no RGB frames found — check data/diving48/rgb/<vid_name>/')

    def _detect_ssv2(self):
        return os.path.exists(os.path.join(self.root_dir, 'labels', 'labels.json'))

    def _preprocess_ssv2(self):
        import json, csv
        with open(os.path.join(self.root_dir, 'labels', 'labels.json')) as f:
            json.load(f)

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
        process_split(train_entries, 'train')

        with open(os.path.join(self.root_dir, 'labels', 'validation.json')) as f:
            val_ann = json.load(f)
        val_entries = [(e['id'], e['template']) for e in val_ann]
        process_split(val_entries, 'val')

        test_entries = []
        with open(os.path.join(self.root_dir, 'labels', 'test-answers.csv')) as f:
            for row in csv.reader(f, delimiter=';'):
                if len(row) >= 2:
                    test_entries.append((row[0], row[1]))
        process_split(test_entries, 'test')

    def _extract_video_to_dir(self, src_path, target_dir):
        _video_preprocess_job((src_path, target_dir, self.clip_len, self.resize_height, self.resize_width))

    def process_video(self, video, action_name, save_dir):
        src_path = os.path.join(self.root_dir, action_name, video)
        video_filename = os.path.splitext(os.path.basename(video))[0]
        target_dir = os.path.join(save_dir, video_filename)
        self._extract_video_to_dir(src_path, target_dir)

    def _compute_and_save_flow(self, video_dir):
        frame_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')])
        if len(frame_paths) < 2:
            return
        imgs = [cv2.imread(p) for p in frame_paths]
        imgs = [img for img in imgs if img is not None]
        if len(imgs) < 2:
            return
        video_tensor = torch.from_numpy(np.stack(imgs, axis=0)).permute(3, 0, 1, 2).float()
        flow = calculate_video_flow(video_tensor)
        np.save(os.path.join(video_dir, "flow.npy"), flow.numpy())

    def ensure_flows(self):
        missing = [d for d in self.fnames if not os.path.exists(os.path.join(d, "flow.npy"))]
        if not missing:
            return
        print(f"==> Computing optical flow for {len(missing)} videos (one-time)...")
        for video_dir in tqdm(missing, desc="Flow"):
            self._compute_and_save_flow(video_dir)

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i in range(len(buffer)):
                buffer[i] = cv2.flip(buffer[i], flipCode=1)
        return buffer

    def color_jitter(self, buffer, brightness=0.3, contrast=0.3):
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        beta = np.random.uniform(-brightness, brightness) * 255.0
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

        # Dynamically sample image properties to prevent shape mismatch on variable test views
        first_img = cv2.imread(frames[0])
        h, w = (first_img.shape[0], first_img.shape[1]) if first_img is not None else (self.resize_height, self.resize_width)

        buffer = np.empty((frame_count, h, w, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.float32)
            else:
                frame = np.array(frame).astype(np.float64)
            buffer[i] = frame
        return buffer

    def ensure_clip_len(self, buffer, clip_len):
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
        max_time = buffer.shape[0] - clip_len
        time_index = np.random.randint(max_time + 1) if max_time > 0 else 0
        height_index = np.random.randint(max(1, buffer.shape[1] - crop_size))
        width_index = np.random.randint(max(1, buffer.shape[2] - crop_size))
        return buffer[time_index:time_index + clip_len, height_index:height_index + crop_size, width_index:width_index + crop_size, :]

    def center_crop(self, buffer, clip_len, crop_size):
        time_index = max(0, (buffer.shape[0] - clip_len) // 2)
        height_index = max(0, (buffer.shape[1] - crop_size) // 2)
        width_index = max(0, (buffer.shape[2] - crop_size) // 2)
        return buffer[time_index:time_index + clip_len, height_index:height_index + crop_size, width_index:width_index + crop_size, :]


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