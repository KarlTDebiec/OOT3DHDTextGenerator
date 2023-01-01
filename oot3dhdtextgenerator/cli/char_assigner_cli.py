#!/usr/bin/env python
#  Copyright 2020-2023 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Character assigner command-line interface."""
from argparse import ArgumentParser

from oot3dhdtextgenerator.common import (
    CommandLineInterface,
    get_arg_groups_by_name,
    input_file_arg,
    int_arg,
    output_file_arg,
    set_logging_verbosity,
)
from oot3dhdtextgenerator.utilities import CharAssigner


class CharAssignerCli(CommandLineInterface):
    """Character assigner command-line interface."""

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
            optional_arguments_name="additional arguments",
        )

        # Input arguments
        arg_groups["input arguments"].add_argument(
            "--n_chars",
            type=int_arg(min_value=10, max_value=9933),
            default=1000,
            help="number of characters included in model, starting from the most "
            "common and ending with the least common (default: %(default)d, max: 9933)",
        )
        arg_groups["input arguments"].add_argument(
            "--assignment-file",
            type=input_file_arg(),
            default="assignment.h5",
            help="assignment input and output file (default: %(default)s)",
        )
        arg_groups["input arguments"].add_argument(
            "--model-infile",
            type=output_file_arg(),
            default="model_1000.pth",
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
    def main(cls) -> None:
        """Execute from command line."""
        parser = cls.argparser()
        kwargs = vars(parser.parse_args())
        set_logging_verbosity(kwargs.pop("verbosity", 1))
        CharAssigner.run(**kwargs)


if __name__ == "__main__":
    CharAssignerCli.main()
