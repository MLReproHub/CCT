import math
from unittest import TestCase

import torch

from layer import PositionalEncoding, PrependClassToken, Patchify


class TestPatchify(TestCase):
    def setUp(self) -> None:
        self.batch_size = 10
        self.patch_size = 16
        self.patch_stride = None
        self.patchifier = Patchify(patch_size=self.patch_size, patch_stride=self.patch_stride)

    def test_forward(self):
        for ps in [8, 16, 32]:
            self.patch_size = self.patch_stride = ps
            self.patchifier.patch_size = ps
            self.patchifier.patch_stride = ps

            x = torch.randn(self.batch_size, 3, 32, 32)
            x_pf = self.patchifier(x.clone())
            # check output shapes
            num_patches = math.floor(
                (32 - self.patchifier.patch_size) / self.patchifier.patch_stride + 1
            ) ** 2
            self.assertListEqual(list(x_pf.shape), [self.batch_size, num_patches, 3, self.patch_size, self.patch_size])
            self.assertEqual(num_patches, self.patchifier.num_patches)
            # check values
            self.assertTrue(torch.equal(
                x_pf[0, 0],
                x[0, :, :self.patch_size, :self.patch_size]
            ))
            self.assertTrue(torch.equal(
                x_pf[0, -1],
                x[0, :, -self.patch_size:, -self.patch_size:]
            ))


class TestPositionalEncoding(TestCase):
    def setUp(self) -> None:
        self.batch_size = 10
        self.d_model = 512
        self.max_len = 50
        self.encoder = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)

    def test_forward(self):
        x = torch.rand(self.batch_size, 4, self.d_model)
        x_with_pe = self.encoder(x)
        self.assertListEqual(list(x_with_pe.shape), list(x.shape))

    def test_encoding(self):
        # Check shapes
        encoding = self.encoder.encoding
        self.assertListEqual(list(encoding.shape), [1, self.max_len, self.d_model])
        # Check values
        encoding_even = encoding[:, :, 0::2]
        encoding_odd = encoding[:, :, 1::2]
        self.assertEqual(torch.isclose(
            encoding_even,
            encoding_odd
        ).sum().item(), 0)
        self.assertTrue(torch.allclose(
            encoding_even ** 2 + encoding_odd ** 2,
            torch.ones_like(encoding_even)
        ))


class TestPrependClassToken(TestCase):
    def setUp(self) -> None:
        self.batch_size = 10
        self.d_model = 512
        self.max_len = 50
        self.clt_prepender = PrependClassToken(d_model=self.d_model)

    def test_forward(self):
        seq_len = self.max_len - 1
        x = torch.rand(self.batch_size, seq_len, self.d_model)
        x_with_clt = self.clt_prepender(x.clone())
        # check output shapes
        self.assertListEqual(list(x_with_clt.shape), [self.batch_size, seq_len + 1, self.d_model])
        # check output values
        self.assertTrue(torch.equal(x_with_clt[:, 1:, :], x))
        self.assertFalse(torch.equal(x_with_clt[:, :-1, :], x))
        self.assertTrue(torch.equal(
            x_with_clt[0, 0, :],
            x_with_clt[1, 0, :]
        ))
        self.assertTrue(torch.equal(
            x_with_clt[0, 0, :],
            x_with_clt[-1, 0, :]
        ))
