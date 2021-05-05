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
          np.asarray(a.shape)*expand_rate))


if __name__ == '__main__':
    unittest.main()


