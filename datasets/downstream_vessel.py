import os
import cv2
import numpy as np
import torch
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets.utils import analyze_name, build_transform, remove_black_edge
from sklearn.model_selection import KFold
from torchvision.transforms import functional as F
import pandas as pd
import albumentations as A


os.environ["OPENCV_LOG_LEVEL"] = "0"


class Downstream_Vessel(Dataset):
    def __init__(self, args, split):
        super(Downstream_Vessel, self).__init__()
        self.args = args
        self.x, self.y, self.names = self.load_name(split)
        assert len(self.x) == len(self.y) == len(self.names)
        self.dataset_size = len(self.x)
        self.train = True if split == "train" else False
        self.im_transform, self.label_transform = build_transform(args, self.train)

    def __len__(self):
        return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
            return idx

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # idx = self._get_index(idx)

        # BGR -> RGB -> PIL
        image = cv2.imread(self.x[idx])
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {self.x[idx]}")
        image = image[..., ::-1]
        image, ymin, ymax, xmin, xmax = remove_black_edge(image)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
        # label
        label = cv2.imread(self.y[idx])
        if label is None:
            raise FileNotFoundError(f"Label not found or unreadable: {self.y[idx]}")
        label = label[ymin: ymax, xmin: xmax]
        label = cv2.resize(label, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        label = label[..., 0]
        label = label // 255

        name = self.names[idx]

        im = Image.fromarray(np.uint8(image))
        target = Image.fromarray(np.uint8(label))

        if self.im_transform is not None:
            # Albumentations pipeline expects dict input
            if isinstance(self.im_transform, A.Compose):
                transformed = self.im_transform(image=np.array(im), mask=np.array(target))
                im_t = transformed['image']
                mask_tensor = transformed['mask']
                # Avoid re-wrapping tensors to silence torch copy warnings
                if torch.is_tensor(mask_tensor):
                    target_t = mask_tensor.long()
                else:
                    target_t = torch.as_tensor(mask_tensor).long()
            else:
                # Keep torchvision-style transforms with shared seed
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                random.seed(seed)
                im_t = self.im_transform(im)

                torch.manual_seed(seed)
                random.seed(seed)
                target_t = self.label_transform(target)
                target_t = F.pil_to_tensor(target_t)
                target_t = torch.squeeze(target_t).long()

        # Debug
        # import imageio
        # im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t.numpy()) * 127
        # imageio.imwrite('/workspace/wangzhonghua/debug/{}.png'.format(name), np.array(im_np).astype(np.uint8))
        # imageio.imwrite('/workspace/wangzhonghua/debug/{}_gt.png'.format(name), np.array(target_np).astype(np.uint8))

        if self.train:
            return im_t, target_t
        else:
            return im_t, target_t, name

    def read_images(self, root_dir):
        image_paths = []
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif']
        for pattern in patterns:
            full_pattern = os.path.join(root_dir, '**', pattern)
            found_paths = glob.glob(full_pattern, recursive=True)
            image_paths.extend(found_paths)
        return image_paths

    def load_name(self, split):
        inputs, targets, names = [], [], []
        dataset_name = getattr(self.args, 'dataset_name', self.args.dataset)
        base_dir = '/workspace/wangzhonghua/fundus_dataset/fundus_miccai'

        # CSV-driven splits for train/val/test
        if split in ['train', 'val', 'test']:
            csv_name = getattr(self.args, 'train_csv_name', 'train.csv') if split == 'train' else f'{split}.csv'
            csv_path = os.path.join(base_dir, "vessel_seg", dataset_name, csv_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f'{split} CSV not found: {csv_path}')

            df = pd.read_csv(csv_path)
            image_col = next((c for c in ['image_path', 'img_path', 'image', 'img'] if c in df.columns), None)
            label_col = next((c for c in ['label_path', 'mask_path', 'label', 'mask'] if c in df.columns), None)
            name_col = 'name' if 'name' in df.columns else None
            if image_col is None or label_col is None:
                raise ValueError(f'CSV {csv_path} must contain image/mask columns, got {df.columns.tolist()}')

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Loading {split} from {dataset_name}'):
                image_path = str(row[image_col])
                target_path = str(row[label_col])

                # Normalize paths: replace /datasets prefix with workspace location
                old_prefix = '/datasets'
                if image_path.startswith(old_prefix):
                    image_path = image_path.replace(old_prefix, base_dir, 1)
                if target_path.startswith(old_prefix):
                    target_path = target_path.replace(old_prefix, base_dir, 1)

                # Enforce dataset filter to avoid mixing other datasets (e.g., DRIVE) by accident
                norm_image = os.path.normpath(image_path)
                norm_target = os.path.normpath(target_path)
                if dataset_name not in norm_image.split(os.sep) or dataset_name not in norm_target.split(os.sep):
                    print(f'=> Skip non-{dataset_name} entry: {image_path}')
                    continue

                if not os.path.exists(image_path) or not os.path.exists(target_path):
                    print(f'=> Missing file, skip: {image_path}, {target_path}')
                    continue
                inputs.append(image_path)
                targets.append(target_path)
                if name_col:
                    names.append(str(row[name_col]))
                else:
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    names.append(f'{dataset_name}_{base}')

            inputs = np.array(inputs)
            targets = np.array(targets)
            names = np.array(names)
            print(f'=> Using {len(inputs)} images from CSV for {split} (dataset: {dataset_name})')
            return inputs, targets, names

        raise ValueError(f'Unsupported split: {split}')


def load_dataset(args, train=False):
    if train:
        train_dataset = Downstream_Vessel(args, 'train')
        val_dataset = Downstream_Vessel(args, 'val')
        return train_dataset, val_dataset
    else:
        test_dataset = Downstream_Vessel(args, 'test')
        return test_dataset


def test():
    label = Image.open('/workspace/julie/datasets/fundus_images/result_sijin/AMD-Training400/mask/A0001.png')[..., ::-1]
    label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label = label[..., 0]
    print(label.shape)
