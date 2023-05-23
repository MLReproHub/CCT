import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from torch import nn, Tensor
from torch.nn import functional as f


class Patchify(nn.Module):
    """
    Patchify Class:
    Splits input image (B, C, H, W) into N patches of size PxP (B, num_patches, C, P, P).
    """

    def __init__(self, patch_size: int = 16, patch_stride: int or None = None):
        super().__init__()
        self._patch_size = patch_size
        self._patch_stride = patch_stride if patch_stride is not None else patch_size
        self.num_patches = math.floor((32 - self.patch_size) / self.patch_stride + 1) ** 2

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def patch_stride(self) -> int:
        return self._patch_stride

    @patch_size.setter
    def patch_size(self, ps: int) -> None:
        self._patch_size = ps
        self.num_patches = math.floor((32 - self.patch_size) / self.patch_stride + 1) ** 2

    @patch_stride.setter
    def patch_stride(self, ps: int) -> None:
        self._patch_stride = ps
        self.num_patches = math.floor((32 - self.patch_size) / self.patch_stride + 1) ** 2

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        :param imgs: shape (B, C, W, H)
        :return: shape (B, num_patches, C, patch_size, patch_size)
        """
        patches = imgs.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
        return patches.reshape(imgs.shape[0], 3, -1, self.patch_size, self.patch_size).swapaxes(1, 2)

    def plot(self, x):
        if x.dim() == 5:
            x = x[0]
        fig = plt.figure()
        outer = gridspec.GridSpec(2, 1)
        ax = plt.Subplot(fig, outer[0])
        ax.axis(False)
        ax.imshow(batch[0].permute(1, 2, 0) * 0.5 + 0.5)
        fig.add_subplot(ax)
        lim = 2 if self.patch_size == 16 else 4
        inner = gridspec.GridSpecFromSubplotSpec(lim, lim, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        for i in range(lim):
            for j in range(lim):
                ax = plt.Subplot(fig, inner[lim * i + j])
                ax.axis(False)
                ax.imshow(x[lim * i + j, :].squeeze().permute(1, 2, 0) * 0.5 + 0.5)
                fig.add_subplot(ax)
        plt.tight_layout()
        plt.savefig('pf.pdf')
        plt.show()


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding Class:
    1D positional sequence encoding based on sine and cosine waves.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Source for encoding: "Attention is all you need", Vaswani et al.
    """

    def __init__(self, d_model: int, max_len: int, dropout_p: float or None = None):
        super().__init__()
        # 1) Create sinusoidal encoding
        # Create position vector and reshape to column vector.
        pos = torch.arange(max_len).unsqueeze(1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        # Encoding for each input in sequence should be same dimension as input, so that they may be added.
        encoding = torch.zeros(1, max_len, d_model)
        # Set even dimensions to sin function
        encoding[0, :, 0::2] = torch.sin(pos / denominator)
        # Set odd dimensions to cos function
        encoding[0, :, 1::2] = torch.cos(pos / denominator)
        # Save as non-trainable parameters
        self.register_buffer('encoding', encoding)

        # 2) Create dropout layer
        self.use_dropout = False
        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
            self.use_dropout = True

    def forward(self, x: Tensor) -> Tensor:
        """
        :param Tensor x: shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.encoding[0, :x.shape[1]]
        if self.use_dropout:
            return self.dropout(x)
        return x

    def plot(self):
        fig, ax = plt.subplots()
        # len x dim
        ax.imshow(self.encoding[0, :, :])
        ax.set_xlabel('dim')
        ax.set_ylabel('seq')
        plt.title('Sinusoidal positional encoding')
        plt.savefig('pe.pdf')
        plt.show()


class PrependClassToken(nn.Module):
    """
    PrependClassToken Class:
    Module to prepend a learnable class token to every sequence in batch.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # The class token does not carry any information in itself. The hidden state corresponding to this token at the
        # end of the transformer will be inferred by all other tokens in the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: dimensions [batch_size, seq_len, embedding_dim]
        :return: x prepended with class token [batch_size, seq_len+1, embedding_dim]
        """
        return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)


# https://arxiv.org/abs/2104.05704
class SequencePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.g = nn.Linear(d_model, 1)

    def forward(self, x):
        x_prime = f.softmax(self.g(x).swapaxes(1, 2), dim=-1)
        # bmm: (bnm) * (bmp) = (bnp)
        # bmm: xl':(b1d) * xl:(bnd) = (b1d)
        z = torch.bmm(x_prime, x)
        return torch.squeeze(z, dim=1)


if __name__ == '__main__':
    from dataset import Cifar10Dataloader

    dl = Cifar10Dataloader(train=True, batch_size=10, shuffle=False)
    batch = next(iter(dl))[0]

    # Plot Patchify
    pf = Patchify(patch_size=8)
    batch_p = pf(batch)
    print(batch_p.shape, np.prod(batch_p.shape[-3:]))
    pf.plot(batch_p)

    # Plot PE
    pe = PositionalEncoding(d_model=256, max_len=100)
    pe.plot()
