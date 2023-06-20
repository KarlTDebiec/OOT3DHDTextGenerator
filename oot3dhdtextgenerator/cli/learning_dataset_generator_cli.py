#!/usr/bin/env python
#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Learning dataset generator command-line interface."""
from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Type

from pipescaler.core.cli import UtilityCli

from oot3dhdtextgenerator.common import (
    float_arg,
    get_arg_groups_by_name,
    int_arg,
    output_file_arg,
    validate_output_file,
)
from oot3dhdtextgenerator.utilities import LearningDatasetGenerator

# TODO: Expose settings for image font, sizes, offsets, fills, and rotations


class LearningDatasetGeneratorCli(UtilityCli):
    """Learning dataset generator command-line interface."""

    @classmethod
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
            "--n_chars",
            type=int_arg(min_value=10, max_value=9933),
            default=9933,
            help="number of unique hanzi to include in dataset, starting from the most "
            "common and ending with the least common (default: %(default)d, max: 9933)",
        )

        # Operation arguments
        arg_groups["operation arguments"].add_argument(
            "--test_proportion",
            default=0.1,
            type=float_arg(min_value=0, max_value=1),
            help="proportion of dataset to be set aside for testing "
            "(default: %(default)f)",
        )

        # Output arguments
        arg_groups["output arguments"].add_argument(
            "--train_outfile",
            type=output_file_arg(),
            default="train_{n_chars}.h5",
            help="train output file (default: %(default)s)",
        )
        arg_groups["output arguments"].add_argument(
            "--test_outfile",
            type=output_file_arg(),
            default="test_{n_chars}.h5",
            help="test output file (default: %(default)s)",
        )

    @classmethod
    def main_internal(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments.

        May be overridden to distribute keyword arguments between initialization of the
        utility and the call to its run method.

        Arguments:
            **kwargs: Keyword arguments
        """
        utility_cls = cls.utility()
        kwargs["train_outfile"] = validate_output_file(
            str(kwargs["train_outfile"]).format(**kwargs)
        )
        kwargs["test_outfile"] = validate_output_file(
            str(kwargs["test_outfile"]).format(**kwargs)
        )
        utility_cls.run(**kwargs)

    @classmethod
    def utility(cls) -> Type[LearningDatasetGenerator]:
        """Type of utility wrapped by command-line interface."""
        return LearningDatasetGenerator


if __name__ == "__main__":
    LearningDatasetGeneratorCli.main()
