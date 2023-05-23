"""
The encompassing ViT model to be trained.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import yaml

import transformer
from layer import Patchify, PositionalEncoding, PrependClassToken, SequencePooling


class ViT(nn.Module):
    def __init__(
            self,
            *,
            kernel_size: int,
            d_model: int,
            h_dim_mlp: int,
            num_heads: int,
            num_enc_layers: int,
            num_classes: int,
            dropout_p: float = 0.1,
            layer_cls=transformer.TransformerEncoderLayer):
        super().__init__()
        self.trained = False

        # Transformer Encoder's Input
        pf = Patchify(kernel_size)
        self.tokenizer = nn.Sequential(
            OrderedDict([
                ("patchify", pf),
                ("flatten", nn.Flatten(start_dim=2)),
                ("patch_projection", nn.Linear(3 * kernel_size ** 2, d_model)),
            ])
        )
        self.cls_token_prepend = PrependClassToken(d_model=d_model)
        # Set max length to num_patches+1 to account for class token
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=500)  # account for padding in CCT

        # Transformer Encoder
        if layer_cls == nn.TransformerEncoderLayer:  # PyTorch implementation
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=h_dim_mlp,
                    dropout=dropout_p,
                    activation='gelu',
                    batch_first=True),
                num_enc_layers)
        elif layer_cls == transformer.TransformerEncoderLayer:  # Own implementation
            self.transformer = transformer.TransformerEncoder(
                d_model=d_model,
                num_heads=num_heads,
                h_dim_mlp=h_dim_mlp,
                activation=nn.GELU,
                dropout_p=dropout_p,
                norm_first=True,
                num_layers=num_enc_layers,
            )

        # Transformer Encoder's Output
        # The hidden layer of the classification head was only used in pre-training. For training from skratch on small
        # datasets only a single linear layer is used https://doi.org/10.48550/arXiv.2104.05704
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = self.cls_token_prepend(tokens)
        tokens = self.pos_enc(tokens)
        z = self.transformer(tokens)
        # Attach classifier to the hidden state corresponding to the class token.
        s = self.classifier(z[:, 0])
        return s


class CVT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_pool = SequencePooling(d_model=kwargs['d_model'])

    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = self.pos_enc(tokens)
        xl = self.transformer(tokens)
        z = self.seq_pool(xl)
        s = self.classifier(z)
        return s


class CCT(CVT):
    def __init__(self, *args, **kwargs):
        self.num_conv_blocks = kwargs.pop('num_conv_blocks')
        conv_padding = kwargs.pop('conv_padding', 0)
        pool_stride = kwargs.pop('pool_stride', 2)
        pool_padding = kwargs.pop('pool_padding', 1)
        self.use_pos_enc = kwargs.pop('use_pos_enc', False)
        super().__init__(*args, **kwargs)

        # Create conv block
        n_channels = [3, ] + [64, ] * (self.num_conv_blocks - 1) + [kwargs['d_model'], ]
        conv_layers = []
        for nc_in, nc_out in zip(n_channels[:-1], n_channels[1:]):
            conv_layers.extend([
                nn.Conv2d(nc_in, nc_out, kernel_size=kwargs['kernel_size'], stride=1, padding=conv_padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=kwargs['kernel_size'], stride=pool_stride, padding=pool_padding),
            ])
        conv_layers.append(nn.Flatten(start_dim=2))
        self.conv_blocks = nn.Sequential(*conv_layers)

    def forward(self, x):
        tokens = self.conv_blocks(x).swapaxes(1, 2)
        if self.use_pos_enc:
            tokens = self.pos_enc(tokens)
        xl = self.transformer(tokens)
        z = self.seq_pool(xl)
        s = self.classifier(z)
        return s


if __name__ == '__main__':
    # cct = CCT(**{
    #     'd_model': 768,
    #     'h_dim_mlp': 3072,
    #     'num_classes': 10,
    #     'num_heads': 12,
    #     'num_enc_layers': 12,
    #     'kernel_size': 3,
    #     'dropout_p': 0.1,
    #     'num_conv_blocks': 2,
    # })
    # _x = torch.rand(2, 3, 32, 32)
    # _cy = cct.conv_blocks(_x)
    # print(_cy.shape)

    from utils import get_total_params

    with open('../configs/cifar10_ViT_Lite_7-16.yaml') as yaml_fp:
        config = yaml.load(yaml_fp, Loader=yaml.FullLoader)
    vit = globals()[config['model']['which']](**config['model']['params'])
    get_total_params(vit, print_table=True, sort_desc=True)
