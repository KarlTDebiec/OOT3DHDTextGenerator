#!/usr/bin/env python
#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from oot3dhdtextgenerator.apps import CharAssigner
from oot3dhdtextgenerator.common import CommandLineInterface
from oot3dhdtextgenerator.common.argument_parsing import (
    get_arg_groups_by_name,
    input_file_arg,
    int_arg,
)
from oot3dhdtextgenerator.common.validation import val_input_path

if TYPE_CHECKING:
    from argparse import ArgumentParser


class CharAssignerCli(CommandLineInterface):
    """Character assigner command-line interface."""

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
            optional_arguments_name="additional arguments",
        )

        # Input arguments
        arg_groups["input arguments"].add_argument(
            "--n-chars",
            type=int_arg(min_value=10, max_value=9933),
            default=9933,
            help="number of characters included in model, starting from the most "
            "common and ending with the least common (default: %(default)d, max: 9933)",
        )
        arg_groups["input arguments"].add_argument(
            "--assignment-file",
            dest="assignment_path",
            type=input_file_arg(),
            default="assignment.h5",
            help="assignment input and output file (default: %(default)s)",
        )
        arg_groups["input arguments"].add_argument(
            "--model-input-file",
            dest="model_input_path",
            type=str,
            default="model_{n_chars}.pth",
            help="model input file (default: %(default)s)",
        )

        # Operation arguments
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

    @classmethod
    @override
    def _main(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments."""
        kwargs["model_input_path"] = val_input_path(
            str(kwargs["model_input_path"]).format(**kwargs)
        )
        char_assigner = CharAssigner(**kwargs)
        char_assigner.run(port=5001)


if __name__ == "__main__":
    CharAssignerCli.main()
