
# coding=utf-8
# The mean of network pruning is inspired by the following repository:
#       https://github.com/huggingface/block_movement_pruning

# Here is the original license of this repository:
# Copyright 2020-present, the HuggingFace Inc. team.
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

"""
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.common_types import _size_2_t
from .constant import *

from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer

sparse_patterns = None

class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
        mask_block_rows: int = 1,
        mask_block_cols: int = 1
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method
        self.mask_block_rows = mask_block_rows
        self.mask_block_cols = mask_block_cols
        self.mask_scores = None

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            size = self.weight.size()
            assert(size[0] % self.mask_block_rows == 0)
            assert(size[1] % self.mask_block_cols == 0)
            mask_size = (size[0] // self.mask_block_rows, size[1] // self.mask_block_cols)
            self.mask_scores = nn.Parameter(torch.Tensor(size=mask_size))
            self.init_mask()

    def init_mask(self):
        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    @staticmethod
    def expand_mask_(mask, mask_block_rows, mask_block_cols):
        mask = torch.repeat_interleave(mask, mask_block_rows, dim=0)
        mask = torch.repeat_interleave(mask, mask_block_cols, dim=1)
        return mask

    @staticmethod
    def check_name(name):
        return name.endswith(".ampere_permut_scores") or name.endswith(".mask_scores")

    @staticmethod
    def mask_(weight, pruning_method, threshold, mask_scores, mask_block_rows, mask_block_cols, training):
        if pruning_method == "topK":
            mask = TopKBinarizer.apply(mask_scores, threshold)
        elif pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in pruning_method
            mask = ThresholdBinarizer.apply(mask_scores, threshold, sig)
        elif pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(weight, threshold)
        elif pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if training:
                u = torch.zeros_like(mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + mask_scores) / b)
            else:
                s = torch.sigmoid(mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        # Expand block mask to individual element mask
        if pruning_method != "magnitude":
            mask = MaskedLinear.expand_mask_(mask,
                                             mask_block_rows=mask_block_rows,
                                             mask_block_cols=mask_block_cols
                                             )

        return mask

    def expand_mask(self, mask):
        return self.expand_mask_(mask, self.mask_block_rows, self.mask_block_cols)

    def forward(self, input: torch.tensor, threshold: float, mask_state: int = OFF):

        mask = self.mask_(self.weight,
                          self.pruning_method,
                          threshold,
                          self.mask_scores,
                          self.mask_block_rows,
                          self.mask_block_cols,
                          training=self.training)

        if mask_state == SOFT_MASK:
            weight_thresholded = mask * self.weight
        elif mask_state == HARD_MASK:
            weight_thresholded = mask * self.weight
            self.weight = torch.nn.Parameter(weight_thresholded)    # change the masked weight
        else:
            weight_thresholded = self.weight

        # Compute output (linear layer) with masked weights
        return F.linear(input, weight_thresholded, self.bias)


class MaskedConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
        mask_block_rows: int = 1,
        mask_block_cols: int = 1
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method
        self.mask_block_rows = mask_block_rows
        self.mask_block_cols = mask_block_cols
        self.mask_scores = None

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            size = self.weight.size()
            assert(size[0] % self.mask_block_rows == 0)
            assert(size[1] % self.mask_block_cols == 0)
            mask_size = (size[0] // self.mask_block_rows, size[1] // self.mask_block_cols)
            self.mask_scores = nn.Parameter(torch.Tensor(size=mask_size))
            self.init_mask()

    def init_mask(self):
        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    @staticmethod
    def expand_mask_(mask, mask_block_rows, mask_block_cols):
        mask = torch.repeat_interleave(mask, mask_block_rows, dim=0)
        mask = torch.repeat_interleave(mask, mask_block_cols, dim=1)
        return mask

    @staticmethod
    def check_name(name):
        return name.endswith(".ampere_permut_scores") or name.endswith(".mask_scores")

    @staticmethod
    def mask_(weight, pruning_method, threshold, mask_scores, mask_block_rows, mask_block_cols, training):
        if pruning_method == "topK":
            mask = TopKBinarizer.apply(mask_scores, threshold)
        elif pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in pruning_method
            mask = ThresholdBinarizer.apply(mask_scores, threshold, sig)
        elif pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(weight, threshold)
        elif pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if training:
                u = torch.zeros_like(mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + mask_scores) / b)
            else:
                s = torch.sigmoid(mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        # Expand block mask to individual element mask
        if pruning_method != "magnitude":
            mask = MaskedLinear.expand_mask_(mask,
                                             mask_block_rows=mask_block_rows,
                                             mask_block_cols=mask_block_cols
                                             )

        return mask

    def forward(self, input: torch.tensor, threshold: float, mask_state: int = OFF):

        mask = self.mask_(self.weight,
                          self.pruning_method,
                          threshold,
                          self.mask_scores,
                          self.mask_block_rows,
                          self.mask_block_cols,
                          training=self.training)

        if mask_state == SOFT_MASK:
            weight_thresholded = mask * self.weight
        elif mask_state == HARD_MASK:
            weight_thresholded = mask * self.weight
            self.weight = torch.nn.Parameter(weight_thresholded)    # change the masked weight
        else:
            weight_thresholded = self.weight
        # Compute output (convolution 2d) with masked weights
        return self._conv_forward(input, weight_thresholded, self.bias)
