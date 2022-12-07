#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model trainer."""
import argparse
from logging import info
from pathlib import Path

import torch
from torch.nn.functional import nll_loss
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.common import (
    CommandLineInterface,
    output_file_arg,
    set_logging_verbosity,
)
from oot3dhdtextgenerator.common.argument_parsing import get_arg_groups_by_name
from oot3dhdtextgenerator.hanzi_dataset import HanziDataset
from oot3dhdtextgenerator.model import Model


class ModelTrainer(CommandLineInterface):
    """Optical character recognition model trainer."""

    @classmethod
    def add_arguments_to_argparser(cls, parser: argparse.ArgumentParser) -> None:
        """Add arguments to a nascent argument parser.

        Arguments:
            parser: Nascent argument parser
        """
        super().add_arguments_to_argparser(parser)

        arg_groups = get_arg_groups_by_name(
            parser,
            "input arguments",
            "operation arguments",
            "output arguments",
            optional_arguments_name="additional arguments",
        )
        arg_groups["operation arguments"].add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help="batch size for training (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--test-batch-size",
            type=int,
            default=1000,
            metavar="N",
            help="batch size for testing (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--epochs",
            type=int,
            default=14,
            metavar="N",
            help="number of epochs to train (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--lr",
            type=float,
            default=1.0,
            metavar="LR",
            help="learning rate (default: %(default)f)",
        )
        arg_groups["operation arguments"].add_argument(
            "--gamma",
            type=float,
            default=0.7,
            metavar="M",
            help="learning rate step gamma (default: %(default)f)",
        )
        arg_groups["operation arguments"].add_argument(
            "--disable-cuda",
            dest="cuda_enabled",
            action="store_false",
            default=True,
            help="disable CUDA",
        )
        arg_groups["operation arguments"].add_argument(
            "--disable-mps",
            dest="mps_enabled",
            action="store_false",
            default=True,
            help="disable macOS GPU",
        )
        arg_groups["operation arguments"].add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="check a single pass",
        )
        arg_groups["operation arguments"].add_argument(
            "--seed",
            type=int,
            default=1,
            metavar="S",
            help="random seed (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--log-interval",
            type=int,
            default=10,
            metavar="N",
            help="training status logging interval (default: %(default)d)",
        )

        arg_groups["output arguments"].add_argument(
            "--outfile",
            type=output_file_arg(),
            default="model.pt",
            help="model output file (default: %(default)s)",
        )

    @classmethod
    def main(cls) -> None:
        """Execute from command line."""
        parser = cls.argparser()
        kwargs = vars(parser.parse_args())
        verbosity = kwargs.pop("verbosity", 1)
        set_logging_verbosity(verbosity)
        cls.main_internal(**kwargs)

    @classmethod
    def main_internal(
        cls,
        *,
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
        outfile: Path,
    ) -> None:
        """Execute from command line.

        Arguments:
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
            outfile: Model output file
        """
        # Determine which device to use
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        mps_enabled = torch.backends.mps.is_available() and mps_enabled
        torch.manual_seed(seed)
        if cuda_enabled:
            device = torch.device("cuda")
        elif mps_enabled:
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

        # dataset1 = MNIST("../data", train=True, download=True, transform=transform)
        dataset1 = HanziDataset("cmn-Hans.h5", transform=transform)
        train_loader = DataLoader(dataset1, **train_loader_kwargs)

        # dataset2 = datasets.MNIST("../data", train=False, transform=transform)
        # test_loader = DataLoader(dataset2, **test_loader_kwargs)

        # Set up model
        model = Model().to(device)
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
            # cls.test(model, device, test_loader)
            scheduler.step()

        # Save model
        torch.save(model.state_dict(), outfile)
        info(f"{cls}: Model saved to {outfile}")

    @staticmethod
    def test(model, device, loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        info(
            f"Test set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(loader.dataset)} "
            f"({100.0 * correct / len(loader.dataset):.0f}%)\n"
        )

    @staticmethod
    def train(
        model,
        device,
        loader,
        optimizer,
        *,
        epoch: int,
        log_interval: int,
        dry_run: bool,
    ):
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


if __name__ == "__main__":
    ModelTrainer.main()
