#!/usr/bin/env python
#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Learning dataset generator command-line interface."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, override

from pipescaler.core.cli import UtilityCli

from oot3dhdtextgenerator.common.argument_parsing import (
    float_arg,
    get_arg_groups_by_name,
    int_arg,
    output_file_arg,
)
from oot3dhdtextgenerator.common.validation import val_output_path
from oot3dhdtextgenerator.utilities import LearningDatasetGenerator

# TODO: Expose settings for image font, sizes, offsets, fills, and rotations


class LearningDatasetGeneratorCli(UtilityCli):
    """Learning dataset generator command-line interface."""

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
            "--train-output-file",
            dest="train_output_path",
            type=output_file_arg(),
            default="train_{n_chars}.h5",
            help="train output file (default: %(default)s)",
        )
        arg_groups["output arguments"].add_argument(
            "--test-output-file",
            dest="test_output_path",
            type=output_file_arg(),
            default="test_{n_chars}.h5",
            help="test output file (default: %(default)s)",
        )

    @classmethod
    @override
    def utility(cls) -> type[LearningDatasetGenerator]:
        """Type of utility wrapped by command-line interface."""
        return LearningDatasetGenerator

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
        kwargs["train_output_path"] = val_output_path(
            str(kwargs["train_output_path"]).format(**kwargs)
        )
        kwargs["test_output_path"] = val_output_path(
            str(kwargs["test_output_path"]).format(**kwargs)
        )
        utility_cls.run(**kwargs)


if __name__ == "__main__":
    LearningDatasetGeneratorCli.main()
