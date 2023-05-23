"""
Data sets and data loaders
"""
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils import data

ASSNMT_DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()


class RandomTranslate:
    def __init__(self, translate_p: float, isize: tuple = (32, 32, 3), srng=None):
        self.iw, self.ih, self.nc = isize
        self.ppc = self.iw * self.ih
        self.srng = srng if srng is not None else np.random.RandomState(42)
        self.translate_p = translate_p

    def __call__(self, x: torch.Tensor, tx=None, ty=None) -> np.ndarray:
        # noinspection PyArgumentList
        if self.srng.rand() >= self.translate_p:
            return x
        if tx is None or ty is None:
            tx = int(self.srng.uniform(-5, 5))
            ty = int(self.srng.uniform(-5, 5))
        ty_abs = np.abs(ty)
        tx_abs = np.abs(tx)
        y_start = (np.sign(ty) - 1) * ty
        x_start = (np.sign(tx) - 1) * tx
        padder = torch.nn.ReflectionPad2d(padding=(tx_abs, tx_abs, ty_abs, ty_abs))
        x_padded = padder(x.unsqueeze(0)).squeeze(0) if x.dim() == 3 else padder(x)
        return x_padded[:, y_start:y_start + self.ih, x_start:x_start + self.iw]


class Cifar10Dataset(tv.datasets.CIFAR10):
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, train: bool = True, hflip_p: float = None, random_crops: bool = False,
                 translate_p: float = None, auto_augment: bool = False):
        img_transforms = []

        if train:
            if hflip_p is not None:
                img_transforms.append(transforms.RandomHorizontalFlip(hflip_p))
            if random_crops:
                # Source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer
                img_transforms.append(transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)))
            if auto_augment:
                img_transforms.extend([
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10,
                                           interpolation=transforms.InterpolationMode.BILINEAR)
                ])

        img_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ])

        # FIX: Random translate operates on Tensors and should not affect normalization. Add as the last transform.
        if train and translate_p is not None:
            img_transforms.append(RandomTranslate(translate_p))

        super().__init__(root=os.path.join(ASSNMT_DIR_PATH, 'data'),
                         train=train,
                         download=True,
                         transform=transforms.Compose(img_transforms),
                         # target_transform=OneHot()
                         )


class Cifar10Dataloader(data.DataLoader):
    def __init__(self, train: bool = True, hflip_p: float = None, random_crops: bool = False,
                 translate_p: float = None, auto_augment: bool = False, **dl_kwargs):
        dataset = Cifar10Dataset(train=train, hflip_p=hflip_p, random_crops=random_crops, translate_p=translate_p,
                                 auto_augment=auto_augment)
        super(Cifar10Dataloader, self).__init__(dataset=dataset, **dl_kwargs)


if __name__ == '__main__':
    ds = Cifar10Dataset()
    ds_tr = Cifar10Dataset(translate_p=1.0)
    sample_idx = 10010

    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(ds[sample_idx][0].permute(1, 2, 0) * 0.5 + 0.5)
    plt.subplot(1, 2, 2)
    plt.imshow(ds_tr[sample_idx][0].permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()
