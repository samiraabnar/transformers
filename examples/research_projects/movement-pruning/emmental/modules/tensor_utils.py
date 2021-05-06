# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensor manipulation utilities."""


import numpy as np
import torch


def expand_tensor(input_tensor: torch.Tensor, expansion_rate: int):
  """ Tiles each element of the tensor locally.

  e.g.:
  input_tensor: [[1,2],
                 [3,4]]
  expansion_rate: 2

  output: [[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]]

  Args:
    input_tensor (`torch.FloatTensor`)
      Input tensor to be expanded.
    expansion_rate (`int`)
      Expansion rate, determined how many times each element in the tensor
      should be repeated in each dimension.
  Returns:
    Expanded tensor (`torch.FloatTensor`)
  """
  shape = input_tensor.shape
  new_shape = [d*expansion_rate for d in shape]
  num_dims = len(shape)
  tiled = input_tensor[...,None].repeat(
    list(np.ones_like(shape)) + [np.power(expansion_rate, num_dims),]
  ).reshape([d for d in shape] + [expansion_rate for d in shape])

  return tiled.permute(
      list(np.reshape(
          [(d, d + len(shape)) for d in range(num_dims)], (-1,)))
      ).reshape(new_shape)


def blockify(input_tensor: torch.Tensor, block_size: int,
             method='max_pool'):
  """Blockify the input tensor.

  e.g.:
  [1, 2, 3, 4] --> [1.5, 1.5, 3.5, 3.5]

  Args:
    input_tensor (`torch.FloatTensor`)
      Input tensor to be blockified.
    block_size (`int`)
      Determines the size of the blocks (in all dimensions).
  Returns:
    Blockified input tensor.
  """
  assert max(np.asarray(input_tensor.shape) % block_size) == 0

  if method == 'max_pool':
    if len(input_tensor.shape) == 1:
      # We add None for batch dimension.
      return expand_tensor(
        torch.nn.MaxPool1d(block_size)(input_tensor[None, ...])[0],
        block_size)
    elif len(input_tensor.shape) == 2:
      # We add None for batch dimension.
      return expand_tensor(
        torch.nn.MaxPool2d(block_size)(input_tensor[None, ...])[0],
        block_size)
    elif len(input_tensor.shape) == 3:
      # We add None for batch dimension.
      return expand_tensor(
        torch.nn.MaxPool3d(block_size)(input_tensor[None, ...])[0],
        block_size)

  raise NotImplementedError