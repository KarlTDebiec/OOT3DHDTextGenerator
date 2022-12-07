#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model."""
from torch import Tensor, flatten
from torch.nn import Conv2d, Dropout, Linear, Module
from torch.nn.functional import log_softmax, max_pool2d, relu


class Model(Module):
    """Optical character recognition model."""

    def __init__(self, n_chars: int) -> None:
        """Initialize.

        Arguments:
            n_chars: Number of characters in dataset
        """
        super().__init__()

        self.conv1 = Conv2d(1, 32, 3, 1)
        self.conv2 = Conv2d(32, 64, 3, 1)
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(2304, 128)
        self.fc2 = Linear(128, n_chars)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Arguments:
            x: Input tensor
        Returns:
            Output tensor
        """
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = log_softmax(x, dim=1)

        return x
