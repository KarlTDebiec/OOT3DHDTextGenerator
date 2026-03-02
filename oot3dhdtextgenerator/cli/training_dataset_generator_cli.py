#!/usr/bin/env python
#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Training dataset generator command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from pipescaler.core.cli import UtilityCli

from oot3dhdtextgenerator.common.argument_parsing import (
    float_arg,
    get_arg_groups_by_name,
    int_arg,
)
from oot3dhdtextgenerator.common.validation import val_output_dir_path, val_output_path
from oot3dhdtextgenerator.core import TrainingDataset
from oot3dhdtextgenerator.data import oot3d_data_path
from oot3dhdtextgenerator.utilities import TrainingDatasetGenerator

if TYPE_CHECKING:
    from argparse import ArgumentParser

# TODO: Expose settings for image font, sizes, offsets, fills, and rotations


class TrainingDatasetGeneratorCli(UtilityCli):
    """Training dataset generator command-line interface."""

    @classmethod
    @override
    def add_arguments_to_argparser(cls, parser: ArgumentParser) -> None:
        """Add arguments to a nascent argument parser.

        Arguments:
            parser: nascent argument parser
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

        # Operation arguments
        arg_groups["operation arguments"].add_argument(
            "--test-proportion",
            default=0.1,
            type=float_arg(min_value=0, max_value=1),
            help="proportion of dataset to be set aside for testing "
            "(default: %(default)f)",
        )

        # Output arguments
        arg_groups["output arguments"].add_argument(
            "--train-output-dir",
            dest="train_output_dir_path",
            type=str,
            default="{oot3d_data_path}/train_{n_chars}",
            help="train output directory (default: %(default)s)",
        )
        arg_groups["output arguments"].add_argument(
            "--test-output-dir",
            dest="test_output_dir_path",
            type=str,
            default="{oot3d_data_path}/test_{n_chars}",
            help="test output directory (default: %(default)s)",
        )
        arg_groups["output arguments"].add_argument(
            "-o",
            "--overwrite",
            action="store_true",
            default=False,
            help="overwrite existing train/test output files",
        )

    @classmethod
    @override
    def utility(cls) -> type[TrainingDatasetGenerator]:
        """Type of utility wrapped by command-line interface."""
        return TrainingDatasetGenerator

    @classmethod
    @override
    def _main(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments.

        May be overridden to distribute keyword arguments between initialization of the
        utility and the call to its run method.

        Arguments:
            **kwargs: keyword arguments
        """
        utility_cls = cls.utility()
        format_kwargs = {
            "n_chars": kwargs["n_chars"],
            "oot3d_data_path": str(oot3d_data_path),
        }
        kwargs["train_output_dir_path"] = val_output_dir_path(
            str(kwargs["train_output_dir_path"]).format(**format_kwargs)
        )
        kwargs["test_output_dir_path"] = val_output_dir_path(
            str(kwargs["test_output_dir_path"]).format(**format_kwargs)
        )
        val_output_path(
            kwargs["train_output_dir_path"] / TrainingDataset.images_npy_file_name,
            exist_ok=kwargs["overwrite"],
        )
        val_output_path(
            kwargs["train_output_dir_path"]
            / TrainingDataset.specifications_csv_file_name,
            exist_ok=kwargs["overwrite"],
        )
        val_output_path(
            kwargs["test_output_dir_path"] / TrainingDataset.images_npy_file_name,
            exist_ok=kwargs["overwrite"],
        )
        val_output_path(
            kwargs["test_output_dir_path"]
            / TrainingDataset.specifications_csv_file_name,
            exist_ok=kwargs["overwrite"],
        )
        kwargs.pop("overwrite")
        utility_cls.run(**kwargs)


if __name__ == "__main__":
    TrainingDatasetGeneratorCli.main()
