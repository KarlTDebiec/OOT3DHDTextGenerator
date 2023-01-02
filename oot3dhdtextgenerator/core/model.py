#!/usr/bin/env python
#  Copyright 2020-2023 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model."""
from __future__ import annotations

from torch import Tensor, flatten, log_softmax, max_pool2d, relu
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

        self.conv1 = Conv2d(1, 32, 3, 1)
        self.conv2 = Conv2d(32, 64, 3, 1)
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(2304, 128)
        self.fc2 = Linear(128, n_chars)
        summary(self.conv1, input_size=(1, 1, 16, 16))
        summary(self.conv2, input_size=(1, 32, 14, 14))
        summary(MaxPool2d(2), input_size=(1, 64, 12, 12))
        summary(Flatten(), input_size=(1, 64, 6, 6))
        summary(self.fc1, input_size=(1, 2304))
        summary(self.fc2, input_size=(1, 128))

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


# def get_model() -> keras.Sequential:
#     model = keras.Sequential(
#         [
#             keras.layers.Conv2D(
#                 filters=64,
#                 kernel_size=2,
#                 padding="same",
#                 activation="relu",
#                 input_shape=(16, 16, 1),
#             ),
#             keras.layers.MaxPooling2D(pool_size=2),
#             keras.layers.Dropout(0.5),
#             keras.layers.Conv2D(
#                 filters=128, kernel_size=2, padding="same", activation="relu"
#             ),
#             keras.layers.MaxPooling2D(pool_size=2),
#             keras.layers.Dropout(0.5),
#             keras.layers.Conv2D(
#                 filters=256, kernel_size=2, padding="same", activation="relu"
#             ),
#             keras.layers.MaxPooling2D(pool_size=2),
#             keras.layers.Dropout(0.5),
#             keras.layers.Flatten(),
#             keras.layers.Dense(n_chars, activation="softmax"),
#         ]
#     )
#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model
