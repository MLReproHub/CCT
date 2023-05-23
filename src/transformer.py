"""
The module containing the transformer block, including the multi-head self-attention block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    """
    MultiHeadSelfAttention Class:
    Defines an MSA layer as proposed in "Attention is all you need paper".
    Note: Ignores masking functionality (and thus cannot be used in Transformer's decoder).
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float or None = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Define projectors
        self.project_q = nn.Linear(d_model, d_model)
        self.project_k = nn.Linear(d_model, d_model)
        self.project_v = nn.Linear(d_model, d_model)
        self.project_out = nn.Linear(d_model, d_model)
        # Define rest components
        self.use_dropout = False
        if dropout_p is not None:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout_p)
        self.dot_scaling = (d_model // num_heads) ** -0.5

    def rearrange_qvk(self, x: Tensor) -> Tensor:
        """
        :param Tensor x: (B, num_seq_samples, d_model)
        :return: Tensor object of shape (B*num_heads, num_seq_samples, d_model//num_heads)
        """
        x = x \
            .reshape(x.shape[0], x.shape[1], self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def rearrange_out(self, x: Tensor) -> Tensor:
        """
        Reverse the operation of `rearrange_qvk`.
        :param Tensor x: (B*num_heads, num_seq_samples, d_model//num_heads)
        :return: Tensor object of shape (B, num_seq_samples, d_model)
        """
        x = x \
            .reshape(-1, self.num_heads, x.shape[1], x.shape[2]) \
            .permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x: Tensor) -> torch.Tensor:
        """
        :param Tensor x: of shape (B, num_seq_samples, d_model)
        :return: Tensor object of shape (B, num_seq_samples, d_model)
        """
        # project QVK
        q, v, k = self.project_q(x), self.project_v(x), self.project_k(x)
        # split QVK to the number of heads
        q, v, k = self.rearrange_qvk(q), self.rearrange_qvk(v), self.rearrange_qvk(k)
        # transpose last two dimensions of key matrix & perform dot products
        # dp = self.dot_scaling * torch.einsum('bhqd, bhkd -> bhqk', q, k)  # batch, num_heads, query_len, key_len
        dp = self.dot_scaling * torch.bmm(q, k.swapaxes(1, 2))
        # perform softmax over the keys --> attention scores
        att = F.softmax(dp, dim=-1)
        if self.use_dropout:
            att = self.dropout(att)
        # sum up over the 3rd axis
        # out = torch.einsum('bhal, bhlv -> bhav ', att, v)torch.bmm(a, v)
        out = torch.bmm(att, v)
        out = self.rearrange_out(out)
        out = self.project_out(out)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer Class:
    Follows the architecture presented in "Attention is all you need".
    """

    def __init__(self, d_model: int, num_heads: int, h_dim_mlp: int, activation=nn.GELU,
                 dropout_p: float or None = 0.1, norm_first: bool = True):
        super(TransformerEncoderLayer, self).__init__()
        self.msa = nn.Sequential(
            MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout_p=None),  # no Dropout in MSA
            nn.Dropout(dropout_p if dropout_p is not None else 0.0)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, h_dim_mlp),
            activation(),
            nn.Dropout(dropout_p if dropout_p is not None else 0.0),
            nn.Linear(h_dim_mlp, d_model),
            nn.Dropout(dropout_p if dropout_p is not None else 0.0)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_first = norm_first

    def forward(self, x: Tensor) -> Tensor:
        # 1) MSA, Add (residual) & Norm
        if self.norm_first:
            x = self.norm1(x)
            x_msa = self.msa(x)
            x = x + x_msa
        else:
            x_msa = self.msa(x)
            x = x + x_msa
            x = self.norm1(x)
        # 2) Feed-forward, Add (residual) & Norm
        if self.norm_first:
            x = self.norm2(x)
            x_ff = self.feed_forward(x)
            x = x + x_ff
        else:
            x_ff = self.feed_forward(x)
            x = x + x_ff
            x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int = 1, **layer_kwargs):
        super(TransformerEncoder, self).__init__()
        self.enc_layers = nn.Sequential(*[
            TransformerEncoderLayer(**layer_kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_layers(x)


if __name__ == '__main__':
    msa = MultiHeadSelfAttention(d_model=768, num_heads=12)
    x = torch.randn(10, 5, 768)
    x_out = msa(x)
    print(x_out.shape)
