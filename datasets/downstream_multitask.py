import os
import cv2
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets.utils import build_transform, remove_black_edge
from torchvision.transforms import functional as F
import pandas as pd
import imageio


os.environ["OPENCV_LOG_LEVEL"] = "0"


class Downstream_Multitask(Dataset):
    """
    Multitask dataset for vessel, odoc, and lesion_ex_he (combined EX/HE).
    - odoc is 3-class: 0 background, 1 disc (mask > 0), 2 cup (mask > 250)
    - vessel is binary (0/1)
    - lesion_ex_he is 3-class: 0 background, 1 EX (mask > 0), 2 HE (mask > 0)
    - data is listed via CSVs under `csvs/<dataset_name>/<split>_relabel.csv`
      with columns: image_path, image_name, label_vessel, label_odoc, label_ex, label_he
    """

    def __init__(self, args, split):
        super(Downstream_Multitask, self).__init__()
        self.args = args
        self.split = split
        self.x, self.y_vessel, self.y_odoc, self.y_ex, self.y_he, self.names = self.load_name(split)
        assert len(self.x) == len(self.y_vessel) == len(self.y_odoc) == len(self.y_ex) == len(self.y_he) == len(self.names)
        self.dataset_size = len(self.x)
        self.train = True if split == "train" else False
        # Use torchvision transforms to keep multi-mask sync simple
        self.im_transform, self.label_transform = build_transform(args, self.train, use_albumentations=False)

    def __len__(self):
        return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
            return idx

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        # Load image
        image = cv2.imread(self.x[idx])[..., ::-1]  # BGR -> RGB
        image, ymin, ymax, xmin, xmax = remove_black_edge(image)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)

        # Vessel (binary)
        label_vessel = cv2.imread(self.y_vessel[idx], cv2.IMREAD_GRAYSCALE)
        label_vessel = label_vessel[ymin:ymax, xmin:xmax]
        label_vessel = cv2.resize(label_vessel, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        label_vessel = (label_vessel > 0).astype(np.uint8)

        # ODoC (3-class: 0 bg, 1 disc, 2 cup)
        label_odoc = cv2.imread(self.y_odoc[idx], cv2.IMREAD_GRAYSCALE)
        label_odoc = label_odoc[ymin:ymax, xmin:xmax]
        label_odoc = cv2.resize(label_odoc, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        disc_mask = label_odoc > 0
        cup_mask = label_odoc > 250  # oc
        label_odoc = np.zeros_like(label_odoc, dtype=np.uint8)
        label_odoc[disc_mask] = 1
        label_odoc[cup_mask] = 2

        # EX (binary)
        label_ex = cv2.imread(self.y_ex[idx], cv2.IMREAD_GRAYSCALE)
        label_ex = label_ex[ymin:ymax, xmin:xmax]
        label_ex = cv2.resize(label_ex, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        label_ex = (label_ex > 0).astype(np.uint8)

        # HE (binary)
        label_he = cv2.imread(self.y_he[idx], cv2.IMREAD_GRAYSCALE)
        label_he = label_he[ymin:ymax, xmin:xmax]
        label_he = cv2.resize(label_he, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        label_he = (label_he > 0).astype(np.uint8)

        # Combine EX/HE into one 3-class mask: 0 bg, 1 EX, 2 HE
        label_ex_he = np.zeros_like(label_ex, dtype=np.uint8)
        label_ex_he[label_ex > 0] = 1
        label_ex_he[label_he > 0] = 2  # HE overrides if overlap
        # Debug masks
        # try:
        #     dbg_dir = '/workspace/wangzhonghua/debug'
        #     os.makedirs(dbg_dir, exist_ok=True)
        #     base_name = os.path.splitext(os.path.basename(self.x[idx]))[0]
        #     imageio.imwrite(os.path.join(dbg_dir, f"{base_name}_vessel.png"), label_vessel.astype(np.uint8) * 255)
        #     imageio.imwrite(os.path.join(dbg_dir, f"{base_name}_odoc.png"), label_odoc.astype(np.uint8) * 127)
        #     imageio.imwrite(os.path.join(dbg_dir, f"{base_name}_ex_he.png"), label_ex_he.astype(np.uint8) * 127)
        # except Exception as e:
        #     print(f"=> Debug mask save failed for {self.x[idx]}: {e}")

        name = self.names[idx]

        # PIL for torchvision transforms
        im = Image.fromarray(np.uint8(image))
        target_vessel = Image.fromarray(label_vessel)
        target_odoc = Image.fromarray(label_odoc)
        target_ex_he = Image.fromarray(label_ex_he)

        # identical transformation for im and all labels
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        im_t = self.im_transform(im) if self.im_transform is not None else im

        torch.manual_seed(seed)
        random.seed(seed)
        target_vessel_t = self.label_transform(target_vessel) if self.label_transform is not None else target_vessel
        target_vessel_t = F.pil_to_tensor(target_vessel_t)
        target_vessel_t = torch.squeeze(target_vessel_t).long()
        target_vessel_t = (target_vessel_t > 0).long()  # enforce {0,1}

        torch.manual_seed(seed)
        random.seed(seed)
        target_odoc_t = self.label_transform(target_odoc) if self.label_transform is not None else target_odoc
        target_odoc_t = F.pil_to_tensor(target_odoc_t)
        target_odoc_t = torch.squeeze(target_odoc_t).long()
        target_odoc_t = torch.clamp(target_odoc_t, 0, 2)  # enforce {0,1,2}

        torch.manual_seed(seed)
        random.seed(seed)
        target_ex_he_t = self.label_transform(target_ex_he) if self.label_transform is not None else target_ex_he
        target_ex_he_t = F.pil_to_tensor(target_ex_he_t)
        target_ex_he_t = torch.squeeze(target_ex_he_t).long()
        target_ex_he_t = torch.clamp(target_ex_he_t, 0, 2)  # enforce {0,1,2}

        if self.train:
            return im_t, [target_vessel_t, target_odoc_t, target_ex_he_t]
        else:
            return im_t, [target_vessel_t, target_odoc_t, target_ex_he_t], name

    def load_name(self, split):
        inputs, targets_vessel, targets_odoc, targets_ex, targets_he, names = [], [], [], [], [], []
        dataset_name = getattr(self.args, 'dataset_name', self.args.dataset)
        base_root = getattr(self.args, 'csv_base_dir', '/workspace/wangzhonghua/fundus_dataset/fundus_miccai')
        # Use test list for both val/test; train uses train list
        if split == 'train':
            split_csv = 'train_relabel.csv'
        else:
            split_csv = 'test_relabel.csv'
        csv_path = os.path.join(base_root, 'lesion_seg', dataset_name, split_csv)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'{split} CSV not found: {csv_path}')

        print(f'=> Loading {split} list from {csv_path}')
        df = pd.read_csv(csv_path)
        required_cols = ['image_path', 'label_vessel', 'label_odoc', 'label_ex', 'label_he']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f'Missing column {col} in {csv_path}')

        # Optional name column
        name_col = 'image_name' if 'image_name' in df.columns else None

        old_prefix = '/datasets'

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Parsing {split} CSV'):
            image_path = str(row['image_path'])
            vessel_path = str(row['label_vessel'])
            odoc_path = str(row['label_odoc'])
            ex_path = str(row['label_ex'])
            he_path = str(row['label_he'])

            # Normalize paths if needed
            if image_path.startswith(old_prefix):
                image_path = image_path.replace(old_prefix, base_root, 1)
            if vessel_path.startswith(old_prefix):
                vessel_path = vessel_path.replace(old_prefix, base_root, 1)
            if odoc_path.startswith(old_prefix):
                odoc_path = odoc_path.replace(old_prefix, base_root, 1)
            if ex_path.startswith(old_prefix):
                ex_path = ex_path.replace(old_prefix, base_root, 1)
            if he_path.startswith(old_prefix):
                he_path = he_path.replace(old_prefix, base_root, 1)

            missing = []
            if not os.path.exists(image_path):
                missing.append(f"image:{image_path}")
            if not os.path.exists(vessel_path):
                missing.append(f"vessel:{vessel_path}")
            if not os.path.exists(odoc_path):
                missing.append(f"odoc:{odoc_path}")
            if not os.path.exists(ex_path):
                missing.append(f"ex:{ex_path}")
            if not os.path.exists(he_path):
                missing.append(f"he:{he_path}")
            if missing:
                print(f'=> Missing file(s), skip: {", ".join(missing)}')
                continue

            inputs.append(image_path)
            targets_vessel.append(vessel_path)
            targets_odoc.append(odoc_path)
            targets_ex.append(ex_path)
            targets_he.append(he_path)
            if name_col:
                names.append(str(row[name_col]))
            else:
                base = os.path.splitext(os.path.basename(image_path))[0]
                names.append(f'{dataset_name}_{base}')

        print(f'=> Using {len(inputs)} images for {split}')
        return np.array(inputs), np.array(targets_vessel), np.array(targets_odoc), np.array(targets_ex), np.array(targets_he), np.array(names)


def load_dataset(args, train=False):
    if train:
        train_dataset = Downstream_Multitask(args, 'train')
        val_dataset = Downstream_Multitask(args, 'val')
        return train_dataset, val_dataset
    else:
        test_dataset = Downstream_Multitask(args, 'test')
        return test_dataset
