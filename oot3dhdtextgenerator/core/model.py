#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model."""

from __future__ import annotations

from torch import Tensor, flatten, log_softmax, relu
from torch.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module
from torchinfo import summary


class Model(Module):
    """Optical character recognition model."""

    def __init__(self, n_chars: int) -> None:
        """Initialize.

        Arguments:
            n_chars: Number of characters in dataset
        """
        super().__init__()

        self.conv1 = Conv2d(1, 64, kernel_size=2, stride=1, padding="same")
        self.conv2 = Conv2d(64, 128, kernel_size=2, stride=1, padding="same")
        self.dropout = Dropout(0.5)
        self.maxpool = MaxPool2d(2)
        self.fc1 = Linear(128 * 4 * 4, n_chars)
        summary(self.conv1, input_size=(1, 1, 16, 16))
        summary(self.maxpool, input_size=(1, 64, 16, 16))
        summary(self.conv2, input_size=(1, 64, 8, 8))
        summary(self.maxpool, input_size=(1, 128, 8, 8))
        summary(Flatten(), input_size=(1, 128, 4, 4))
        summary(self.fc1, input_size=(1, 128 * 4 * 4))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Arguments:
            x: Input tensor
        Returns:
            Output tensor
        """
        x = self.conv1(x)
        x = relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = log_softmax(x, dim=1)

        return x
