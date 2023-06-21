#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for oot3dhdtextgenerator command-line interface."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from inspect import getfile
from io import StringIO
from pathlib import Path
from typing import Type

from pipescaler.testing import parametrize_with_readable_ids

from oot3dhdtextgenerator.cli import (
    CharAssignerCli,
    LearningDatasetGeneratorCli,
    ModelTrainerCli,
)
from oot3dhdtextgenerator.common import CommandLineInterface, run_cli_with_args


@parametrize_with_readable_ids(
    ("command"),
    [
        (CharAssignerCli),
        (LearningDatasetGeneratorCli),
        (ModelTrainerCli),
    ],
)
def test_help(command: Type[CommandLineInterface]) -> None:
    stdout = StringIO()
    stderr = StringIO()
    try:
        with redirect_stdout(stdout):
            with redirect_stderr(stderr):
                run_cli_with_args(command, "-h")
    except SystemExit as error:
        assert error.code == 0
        assert stdout.getvalue().startswith(f"usage: {Path(getfile(command)).name}")
        assert stderr.getvalue() == ""
