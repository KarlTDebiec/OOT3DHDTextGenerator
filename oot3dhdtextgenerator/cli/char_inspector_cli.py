#!/usr/bin/env python
#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character inspector command-line interface."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, override

from oot3dhdtextgenerator.apps import CharInspector
from oot3dhdtextgenerator.common import CommandLineInterface
from oot3dhdtextgenerator.common.argument_parsing import (
    get_arg_groups_by_name,
    int_arg,
)
from oot3dhdtextgenerator.common.validation import val_output_dir_path
from oot3dhdtextgenerator.data import oot3d_data_path


class CharInspectorCli(CommandLineInterface):
    """Character inspector command-line interface."""

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

        arg_groups["input arguments"].add_argument(
            "--n-chars",
            type=int_arg(min_value=10, max_value=9933),
            default=9933,
            help="number of active characters to include (default: %(default)d, max: "
            "9933)",
        )
        arg_groups["input arguments"].add_argument(
            "--assignment-dir",
            dest="assignment_dir_path",
            type=str,
            default="{oot3d_data_path}",
            help=(
                "assignment input and output CSV directory containing assigned.csv "
                "and unassigned.csv (default: %(default)s)"
            ),
        )
        arg_groups["operation arguments"].add_argument(
            "-p",
            "--port",
            type=int_arg(min_value=1, max_value=65535),
            default=5002,
            help="port on which to run character inspector (default: %(default)d)",
        )

    @classmethod
    @override
    def _main(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments."""
        port = kwargs.pop("port", 5002)
        format_kwargs = {
            "oot3d_data_path": str(oot3d_data_path),
        }
        kwargs["assignment_dir_path"] = val_output_dir_path(
            str(kwargs["assignment_dir_path"]).format(**format_kwargs)
        )
        char_inspector = CharInspector(**kwargs)
        char_inspector.run(port=port)


if __name__ == "__main__":
    CharInspectorCli.main()
