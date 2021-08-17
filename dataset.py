import collections
import torch
import PIL
import pytorch_lightning as pl
import numpy as np
import torchvision.transforms.functional as F

from pathlib import Path
from torchvision import transforms as T
from random import randint, choice
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder: str,
                 image_size=224,
                 resize_ratio=0.75,
                 is_eval=False):
        """
        Conditional CLIP dataset.

        Args:
            folder (str): Folder containing images and text
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize.
            is_eval (bool, optional): Whether this is running in evaluation mode. Defaults to False.
        """
        super().__init__()
        self.is_eval = is_eval
        path = Path(folder)

        context_files = [*path.glob('**/*.ctx')]
        target_files = [*path.glob('**/*.tgt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        context_files = {context_file.stem: context_file for context_file in context_files}
        target_files = {target_file.stem: target_file for target_file in target_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & context_files.keys() & target_files.keys())

        self.keys = list(keys)
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.context_files = {k: v for k, v in context_files.items() if k in keys}
        self.target_files = {k: v for k, v in target_files.items() if k in keys}

        self.resize_ratio = resize_ratio
        if self.is_eval:
            self.image_transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.Lambda(self.fix_img),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.image_transform = T.Compose([
                T.Lambda(self.fix_img),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

    def __len__(self):
        return len(self.keys)
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        context_file = self.context_files[key]
        target_file = self.target_files[key]
        image_file = self.image_files[key]

        try:
            contexts = context_file.read_text().split('\n')
            targets = target_file.read_text().split('\n')
        except UnicodeDecodeError:
            return self.skip_sample(ind)
        contexts = list(filter(lambda t: len(t) > 0, contexts))
        targets = list(filter(lambda t: len(t) > 0, targets))
        try:
            idx = choice(range(len(contexts)))
            context = contexts[idx]
            target = targets[idx]
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        return image_tensor, context, target


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_datadir,
                 dev_datadir,
                 batch_size=64,
                 nworkers=0):
        super().__init__()
        self.train_datadir = train_datadir
        self.dev_datadir = dev_datadir
        self.batch_size = batch_size
        self.nworkers = nworkers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = Dataset(folder=self.train_datadir)
            self.valid = Dataset(folder=self.dev_datadir)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nworkers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.nworkers,
            pin_memory=True)