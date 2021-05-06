"""Unit tests for methods defined in tensor_utils.py"""


import numpy as np
import tensor_utils
import torch
import unittest

class TestTensorUtils(unittest.TestCase):

    def test_expand_tensor(self):
      a = torch.Tensor([[1,2,3],[4,5,6]])

      for expand_rate in [1, 2, 3]:
        e_a = tensor_utils.expand_tensor(a, expand_rate)
        self.assertTrue(np.array_equal(
          np.asarray(e_a.shape),
          np.asarray(a.shape) * expand_rate))

    def test_blockify(self):
      a = torch.Tensor([[1,2,3],[4,5,6], [7, 8, 9]])

      e_a = tensor_utils.blockify(a, 1)
      self.assertTrue(np.array_equal(
        e_a,
        a))
      e_a = tensor_utils.blockify(a, 3)
      self.assertTrue(np.array_equal(
        e_a,
        torch.ones_like(a) * torch.max(a)))
      for block_size in [1, 3]:
        e_a = tensor_utils.blockify(a, block_size)
        self.assertTrue(np.array_equal(
          np.asarray(e_a.shape),
          np.asarray(a.shape)))


if __name__ == '__main__':
    unittest.main()


