import os
import cv2
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets.utils import analyze_name, build_transform, remove_black_edge
from torchvision.transforms import functional as F
import pandas as pd
import albumentations as A


os.environ["OPENCV_LOG_LEVEL"] = "0"


class Downstream_Lesion(Dataset):
    def __init__(self, args, split):
        super(Downstream_Lesion, self).__init__()
        self.args = args
        self.x, self.y1, self.y2, self.y3, self.y4, self.names = self.load_name(split)
        assert len(self.x) == len(self.y1) == len(self.y2) == len(self.y3) == len(self.y4) == len(self.names)
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
        image = cv2.imread(self.x[idx])[..., ::-1]
        image, ymin, ymax, xmin, xmax = remove_black_edge(image)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
        
        # Initialize a single label array
        label = np.zeros((self.args.size, self.args.size), dtype=np.uint8)
        
        # Process all labels in a loop
        label_paths = [self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx]]
        for i, label_path in enumerate(label_paths):
            if not label_path or not os.path.exists(label_path):
                print(f"=> Missing label path for class {i+1}: {label_path}")
                continue
            temp_label = cv2.imread(label_path)
            if temp_label is None:
                print(f"=> Failed to load label at path: {label_path}")
                continue
            temp_label[temp_label > 0] = 255  # enforce binary
            temp_label = temp_label[ymin:ymax, xmin:xmax]
            temp_label = cv2.resize(temp_label, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
            # Some label files are single-channel; use the first channel consistently
            if temp_label.ndim == 3:
                temp_label = temp_label[..., 2]
            temp_label = temp_label // 255
            # Add to main label with appropriate weight
            label[temp_label > 0] = i + 1

        name = self.names[idx]

        im = Image.fromarray(np.uint8(image))
        target = Image.fromarray(np.uint8(label))

        # identical transformation for im and gt
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            if isinstance(self.im_transform, A.Compose):
                transformed = self.im_transform(image=np.array(im), mask=np.array(target))
                im_t = transformed["image"]
                mask_tensor = transformed["mask"]
                target_t = mask_tensor.long() if torch.is_tensor(mask_tensor) else torch.as_tensor(mask_tensor).long()
            else:
                im_t = self.im_transform(im)

        if self.label_transform is not None and not isinstance(self.im_transform, A.Compose):
            torch.manual_seed(seed)
            random.seed(seed)
            target_t = self.label_transform(target)
            target_t = F.pil_to_tensor(target_t)
            target_t = torch.squeeze(target_t).long()
        elif not isinstance(self.im_transform, A.Compose):
            target_t = F.pil_to_tensor(target)
            target_t = torch.squeeze(target_t).long()
        # ensure label values within class range
        max_class = max(1, self.args.out_channels - 1) if hasattr(self.args, "out_channels") else 4
        target_t = torch.clamp(target_t, min=0, max=max_class)
        if 'im_t' not in locals():
            im_t = self.im_transform(im) if self.im_transform is not None else F.pil_to_tensor(im)
        if label.sum() == 0:
            print(f"=> Warning: zero label mask for {name}")

        # Debug
        import imageio
        im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        target_np = (target_t.numpy()) * 127
        imageio.imwrite('/workspace/wangzhonghua/debug/{}.png'.format(name), np.array(im_np).astype(np.uint8))
        imageio.imwrite('/workspace/wangzhonghua/debug/{}_gt.png'.format(name), np.array(target_np).astype(np.uint8))

        if self.train:
            return im_t, target_t
        else:
            return im_t, target_t, name

    def load_name(self, split):
        inputs, ex_targets, he_targets, se_targets, ma_targets, names = [], [], [], [], [], []
        base_root = getattr(self.args, "csv_base_dir", "/workspace/wangzhonghua/fundus_dataset/fundus_miccai")
        dataset_dir = os.path.join(base_root, "lesion_seg", self.args.dataset_name)
        if split == 'train':
            csv_name = getattr(self.args, "train_csv_name", "train.csv")
        elif split == 'val':
            csv_name = 'val.csv'
        else:
            csv_name = 'test.csv'
        csv_path = os.path.join(dataset_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Loading {split} CSV'):
            img_path = str(row['image_path'])
            ex_path = str(row['label_EX_path'])
            he_path = str(row['label_HE_path'])
            ma_path = str(row['label_MA_path'])
            se_path = str(row['label_SE_path'])
            name = str(row.get('image_name', analyze_name(img_path)))

            img_path = img_path.replace('/datasets', base_root)
            ex_path = ex_path.replace('/datasets', base_root)
            he_path = he_path.replace('/datasets', base_root)
            ma_path = ma_path.replace('/datasets', base_root)
            se_path = se_path.replace('/datasets', base_root)

            inputs.append(img_path)
            ex_targets.append(ex_path)
            he_targets.append(he_path)
            se_targets.append(se_path)
            ma_targets.append(ma_path)
            names.append(name)

        return np.array(inputs), np.array(he_targets), np.array(ex_targets), np.array(se_targets), np.array(ma_targets), np.array(names)


def load_dataset(args, train=False):
    if train:
        train_dataset = Downstream_Lesion(args, 'train')
        val_dataset = Downstream_Lesion(args, 'val')
        return train_dataset, val_dataset
    else:
        test_dataset = Downstream_Lesion(args, 'test')
        return test_dataset


def test():
    label = Image.open('/workspace/julie/datasets/fundus_images/result_sijin/AMD-Training400/mask/A0001.png')[..., ::-1]
    label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label = label[..., 0]
    print(label.shape)
