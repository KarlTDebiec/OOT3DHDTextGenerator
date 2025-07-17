#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model trainer."""
from __future__ import annotations

from logging import info
from pathlib import Path

import torch
from pipescaler.core import Utility
from torch.nn.functional import nll_loss
from torch.optim import Adadelta, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.core import LearningDataset, Model


class ModelTrainer(Utility):
    """Optical character recognition model trainer."""

    @classmethod
    def run(
        cls,
        *,
        train_infile: Path,
        test_infile: Path,
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
        model_outfile: Path,
    ) -> None:
        """Execute from command line.

        Arguments:
            train_infile: Train data input file
            test_infile: Test data input file
            batch_size: Batch size for training
            test_batch_size: Batch size for testing
            epochs: Number of epochs to train
            lr: Learning rate
            gamma: Learning rate step gamma
            cuda_enabled: Whether to use CUDA
            mps_enabled: Whether to use macOS GPU
            dry_run: Whether to check a single pass
            seed: Random seed
            log_interval: Training status logging interval
            model_outfile: Model output file
        """
        # Determine which device to use
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        torch.manual_seed(seed)
        if cuda_enabled:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Set up training and test settings
        train_loader_kwargs = {"batch_size": batch_size}
        test_loader_kwargs = {"batch_size": test_batch_size}
        if cuda_enabled:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            train_loader_kwargs.update(cuda_kwargs)
            test_loader_kwargs.update(cuda_kwargs)

        # Load training and test data
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        train_dataset = LearningDataset(train_infile, transform=transform)
        train_loader = DataLoader(train_dataset, **train_loader_kwargs)
        test_dataset = LearningDataset(test_infile, transform=transform)
        test_loader = DataLoader(test_dataset, **test_loader_kwargs)

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
        torch.save(model.state_dict(), model_outfile)
        info(f"{cls}: Model saved to {model_outfile}")

    @staticmethod
    def test(model: Model, device: torch.device, loader: DataLoader) -> None:
        """Test model against test data.

        Arguments:
            model: Model to test
            device: Device to use
            loader: Test data loader
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        info(
            f"Test set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(loader.dataset)} "
            f"({100.0 * correct / len(loader.dataset):.0f}%)\n"
        )

    @staticmethod
    def train(
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
            model: Model to train
            device: Device to use
            loader: Training data loader
            optimizer: Optimizer to use
            epoch: Epoch number
            log_interval: Logging interval
            dry_run: Whether to check a single pass
        """
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                info(
                    f"Train epoch {epoch} "
                    f"[{batch_idx * len(data)}/{len(loader.dataset)} "
                    f"({100.0 * batch_idx / len(loader):.0f}%)] "
                    f"Loss: {loss.item():.6f}"
                )
                if dry_run:
                    break
