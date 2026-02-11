#!/usr/bin/env python
#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model trainer command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from pipescaler.core.cli import UtilityCli

from oot3dhdtextgenerator.common.argument_parsing import (
    get_arg_groups_by_name,
    int_arg,
    output_file_arg,
)
from oot3dhdtextgenerator.common.validation import (
    val_input_path,
    val_output_path,
)
from oot3dhdtextgenerator.utilities import ModelTrainer

if TYPE_CHECKING:
    from argparse import ArgumentParser


class ModelTrainerCli(UtilityCli):
    """Optical character recognition model trainer command-line interface."""

    @classmethod
    @override
    def add_arguments_to_argparser(cls, parser: ArgumentParser) -> None:
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

        # Input arguments
        arg_groups["input arguments"].add_argument(
            "--n-chars",
            type=int_arg(min_value=10, max_value=9933),
            default=9933,
            help="number of unique hanzi to include in dataset, starting from the most "
            "common and ending with the least common (default: %(default)d, max: 9933)",
        )
        arg_groups["input arguments"].add_argument(
            "--train-input-file",
            dest="train_input_path",
            type=str,
            default="train_{n_chars}.h5",
            help="train data input file (default: %(default)s)",
        )
        arg_groups["input arguments"].add_argument(
            "--test-input-file",
            dest="test_input_path",
            type=str,
            default="test_{n_chars}.h5",
            help="test data input file (default: %(default)s)",
        )

        # Operation arguments
        arg_groups["operation arguments"].add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="batch size for training (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--test-batch-size",
            type=int,
            default=10000,
            help="batch size for testing (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--epochs",
            type=int,
            default=10,
            help="number of epochs to train (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--lr",
            type=float,
            default=1.0,
            help="learning rate (default: %(default)f)",
        )
        arg_groups["operation arguments"].add_argument(
            "--gamma",
            type=float,
            default=0.7,
            help="learning rate step gamma (default: %(default)f)",
        )
        arg_groups["operation arguments"].add_argument(
            "--seed",
            type=int,
            default=1,
            help="random seed (default: %(default)d)",
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

        # Output arguments
        arg_groups["output arguments"].add_argument(
            "--log-interval",
            type=int,
            default=1,
            help="training status logging interval (default: %(default)d)",
        )
        arg_groups["output arguments"].add_argument(
            "--model-output-file",
            dest="model_output_path",
            type=output_file_arg(),
            default="model_{n_chars}.pth",
            help="model output file (default: %(default)s)",
        )

    @classmethod
    @override
    def utility(cls) -> type[ModelTrainer]:
        """Type of utility wrapped by command-line interface."""
        return ModelTrainer

    @classmethod
    @override
    def _main(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments.

        May be overridden to distribute keyword arguments between initialization of the
        utility and the call to its run method.

        Arguments:
            **kwargs: Keyword arguments
        """
        utility_cls = cls.utility()
        kwargs["train_input_path"] = val_input_path(
            str(kwargs["train_input_path"]).format(**kwargs)
        )
        kwargs["test_input_path"] = val_input_path(
            str(kwargs["test_input_path"]).format(**kwargs)
        )
        kwargs["model_output_path"] = val_output_path(
            str(kwargs["model_output_path"]).format(**kwargs)
        )
        kwargs.pop("n_chars")
        utility_cls.run(**kwargs)


if __name__ == "__main__":
    ModelTrainerCli.main()
