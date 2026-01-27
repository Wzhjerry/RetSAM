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
import albumentations as A


os.environ["OPENCV_LOG_LEVEL"] = "0"


class Downstream_DC(Dataset):
    """
    OD/OC dataset loader using CSV lists.
    - Outputs 3 classes: 0 background, 1 disc (>0), 2 cup (>250)
    - CSV location: /workspace/wangzhonghua/fundus_dataset/fundus_miccai/ODOC_seg/<dataset_name>/<split>.csv
      Columns: image_path, label_path, optional name/image_name
    """

    def __init__(self, args, split):
        super(Downstream_DC, self).__init__()
        self.args = args
        self.split = split
        self.x, self.y, self.names = self.load_name(split)
        assert len(self.x) == len(self.y) == len(self.names)
        self.dataset_size = len(self.x)
        self.train = True if split == "train" else False
        self.im_transform, self.label_transform = build_transform(args, self.train)

    def __len__(self):
        return self.dataset_size

    def _get_index(self, idx):
        return idx % self.dataset_size if self.train else idx

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        # BGR -> RGB
        image = cv2.imread(self.x[idx])[..., ::-1]
        image, ymin, ymax, xmin, xmax = remove_black_edge(image)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)

        # Label: 3 classes (0 bg, 1 disc, 2 cup)
        label = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)
        label = label[ymin: ymax, xmin: xmax]
        label = cv2.resize(label, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        label_disc = label > 0
        label_cup = label > 250
        label = np.zeros_like(label, dtype=np.uint8)
        label[label_disc] = 1
        label[label_cup] = 2

        name = self.names[idx]

        im = Image.fromarray(np.uint8(image))
        target = Image.fromarray(np.uint8(label))

        if self.im_transform is not None:
            if isinstance(self.im_transform, A.Compose):
                transformed = self.im_transform(image=np.array(im), mask=np.array(target))
                im_t = transformed['image']
                mask_tensor = transformed['mask']
                target_t = mask_tensor.long() if torch.is_tensor(mask_tensor) else torch.as_tensor(mask_tensor).long()
            else:
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                random.seed(seed)
                im_t = self.im_transform(im)

                torch.manual_seed(seed)
                random.seed(seed)
                target_t = self.label_transform(target)
                target_t = F.pil_to_tensor(target_t)
                target_t = torch.squeeze(target_t).long()
        else:
            im_t = im
            target_t = torch.from_numpy(np.array(target)).long()

        if self.train:
            return im_t, target_t
        else:
            return im_t, target_t, name

    def load_name(self, split):
        inputs, targets, names = [], [], []
        dataset_name = getattr(self.args, 'dataset_name', self.args.dataset)
        base_dir = '/workspace/wangzhonghua/fundus_dataset/fundus_miccai'
        if split == 'train':
            csv_name = getattr(self.args, 'train_csv_name', 'train.csv')
        else:
            csv_name = f'{split}.csv'
        csv_path = os.path.join(base_dir, 'ODOC_seg', dataset_name, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'{split} CSV not found: {csv_path}')

        print(f'=> Loading {split} list from {csv_path}')
        df = pd.read_csv(csv_path)
        required_cols = ['image_path', 'label_path']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f'Missing column {col} in {csv_path}')

        name_col = 'name' if 'name' in df.columns else 'image_name' if 'image_name' in df.columns else None
        old_prefix = '/datasets'

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Parsing {split} CSV'):
            image_path = str(row['image_path'])
            label_path = str(row['label_path'])
            if image_path.startswith(old_prefix):
                image_path = image_path.replace(old_prefix, base_dir, 1)
            if label_path.startswith(old_prefix):
                label_path = label_path.replace(old_prefix, base_dir, 1)

            missing = []
            if not os.path.exists(image_path):
                missing.append(f"image:{image_path}")
            if not os.path.exists(label_path):
                missing.append(f"label:{label_path}")
            if missing:
                print(f'=> Missing file(s), skip: {", ".join(missing)}')
                continue

            inputs.append(image_path)
            targets.append(label_path)
            if name_col:
                names.append(str(row[name_col]))
            else:
                base = os.path.splitext(os.path.basename(image_path))[0]
                names.append(f'{dataset_name}_{base}')

        print(f'=> Using {len(inputs)} images for {split}')
        return np.array(inputs), np.array(targets), np.array(names)


def load_dataset(args, train=False):
    if train:
        train_dataset = Downstream_DC(args, 'train')
        val_dataset = Downstream_DC(args, 'val')
        return train_dataset, val_dataset
    else:
        test_dataset = Downstream_DC(args, 'test')
        return test_dataset
