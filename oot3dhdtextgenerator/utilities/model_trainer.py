#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model trainer."""

from __future__ import annotations

import random
from collections.abc import Sized
from logging import info
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from pipescaler.core import Utility
from torch.nn.functional import nll_loss
from torch.optim import Adadelta, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.core import Model, TrainingDataset

if TYPE_CHECKING:
    from pathlib import Path


class ModelTrainer(Utility):
    """Optical character recognition model trainer."""

    @classmethod
    def run(  # noqa: PLR0913
        cls,
        *,
        train_input_dir_path: Path,
        test_input_dir_path: Path,
        batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 1,
        lr: float = 1.0,
        gamma: float = 0.7,
        cuda_enabled: bool = True,
        mps_enabled: bool = True,
        dry_run: bool = False,
        seed: int = 1,
        log_interval: int = 10,
        model_output_path: Path,
    ) -> None:
        """Execute from command line.

        Arguments:
            train_input_dir_path: train data input directory
            test_input_dir_path: test data input directory
            batch_size: batch size for training
            test_batch_size: batch size for testing
            epochs: number of epochs to train
            lr: learning rate
            gamma: learning rate step gamma
            cuda_enabled: whether to use CUDA
            mps_enabled: whether to use macOS GPU
            dry_run: whether to check a single pass
            seed: random seed
            log_interval: training status logging interval
            model_output_path: model output file
        """
        # Determine which device to use
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if cuda_enabled:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load training and test data
        train_dataset = TrainingDataset(train_input_dir_path)
        test_dataset = TrainingDataset(test_input_dir_path)
        normalization_mean, normalization_std = cls.get_normalization_stats(
            train_dataset
        )
        transform = Compose(
            [ToTensor(), Normalize((normalization_mean,), (normalization_std,))]
        )
        train_dataset.transform = transform
        test_dataset.transform = transform
        info(
            "Using normalization mean %.6f and std %.6f",
            normalization_mean,
            normalization_std,
        )

        if cuda_enabled:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=1,
            )

        # Configure model
        n_chars = len(set(train_dataset.specifications["character"]))
        model = Model(n_chars).to(device)
        optimizer = Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        # Train model
        for epoch in range(1, epochs + 1):
            cls.train(
                model,
                device,
                train_loader,
                optimizer,
                epoch=epoch,
                log_interval=log_interval,
                dry_run=dry_run,
            )
            cls.test(model, device, test_loader)
            scheduler.step()

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "n_chars": n_chars,
            "normalization": {"mean": normalization_mean, "std": normalization_std},
        }
        torch.save(checkpoint, model_output_path)
        info(f"Model saved to {model_output_path}")

    @staticmethod
    def get_normalization_stats(dataset: TrainingDataset) -> tuple[float, float]:
        """Calculate normalization statistics from the training dataset.

        Arguments:
            dataset: training dataset
        Returns:
            mean and standard deviation in [0, 1] scale
        """
        images = dataset.images.astype(np.float32) / 255.0
        mean = float(images.mean())
        std = float(images.std())
        if std <= 0:
            std = 1.0
        return mean, std

    @staticmethod
    def test(model: Model, device: torch.device, loader: DataLoader) -> None:
        """Test model against test data.

        Arguments:
            model: model to test
            device: device to use
            loader: test data loader
        """
        model.eval()
        test_loss = 0
        correct = 0
        dataset_size = len(cast(Sized, loader.dataset))
        with torch.no_grad():
            for data, target in loader:
                batch_data = data.to(device)
                batch_target = target.to(device)
                output = model(batch_data)
                test_loss += nll_loss(output, batch_target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(batch_target.view_as(pred)).sum().item()
        test_loss /= dataset_size
        info(
            f"Test set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{dataset_size} "
            f"({100.0 * correct / dataset_size:.0f}%)\n"
        )

    @staticmethod
    def train(  # noqa: PLR0913
        model: Model,
        device: torch.device,
        loader: DataLoader,
        optimizer: Optimizer,
        *,
        epoch: int,
        log_interval: int,
        dry_run: bool,
    ):
        """Train model against training data.

        Arguments:
            model: model to train
            device: device to use
            loader: training data loader
            optimizer: optimizer to use
            epoch: epoch number
            log_interval: logging interval
            dry_run: whether to check a single pass
        """
        model.train()
        dataset_size = len(cast(Sized, loader.dataset))
        for batch_idx, (data, target) in enumerate(loader):
            batch_data = data.to(device)
            batch_target = target.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = nll_loss(output, batch_target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                info(
                    f"Train epoch {epoch} "
                    f"[{batch_idx * len(batch_data)}/{dataset_size} "
                    f"({100.0 * batch_idx / len(loader):.0f}%)] "
                    f"Loss: {loss.item():.6f}"
                )
                if dry_run:
                    break
