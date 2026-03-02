#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for oot3dhdtextgenerator command-line interface."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from inspect import getfile
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipescaler.testing.mark import parametrize_with_readable_ids

import oot3dhdtextgenerator.cli.model_trainer_cli as model_trainer_cli_module
from oot3dhdtextgenerator.cli import (
    CharAssignerCli,
    ModelTrainerCli,
    TrainingDatasetGeneratorCli,
)
from oot3dhdtextgenerator.common.testing import run_cli_with_args

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from oot3dhdtextgenerator.common import CommandLineInterface


@parametrize_with_readable_ids(
    ("command"),
    [
        (CharAssignerCli),
        (TrainingDatasetGeneratorCli),
        (ModelTrainerCli),
    ],
)
def test_help(command: type[CommandLineInterface]) -> None:
    """Test command help output."""
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


def test_model_trainer_overwrite_arg() -> None:
    """Test model trainer overwrite argument parsing."""
    parser = ModelTrainerCli.argparser()
    assert parser.parse_args([]).overwrite is False
    assert parser.parse_args(["-o"]).overwrite is True
    assert parser.parse_args(["--overwrite"]).overwrite is True


def test_model_trainer_overwrite_uses_exist_ok(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test model trainer overwrite option is forwarded to output validation."""
    captured: dict[str, Any] = {}

    def fake_val_input_dir_path(value: str) -> Path:
        """Validate mocked input directory path."""
        return Path(value)

    def fake_val_output_path(value: str, exist_ok: bool = False) -> Path:
        """Validate mocked output path."""
        captured["exist_ok"] = exist_ok
        return Path(value)

    class FakeUtility:
        """Mock utility class."""

        @staticmethod
        def run(**kwargs: Any):
            """Capture run keyword arguments."""
            captured["run_kwargs"] = kwargs

    def fake_utility(cls: type[ModelTrainerCli]) -> type[FakeUtility]:
        """Get mocked utility."""
        _ = cls
        return FakeUtility

    monkeypatch.setattr(
        model_trainer_cli_module, "val_input_dir_path", fake_val_input_dir_path
    )
    monkeypatch.setattr(
        model_trainer_cli_module, "val_output_path", fake_val_output_path
    )
    monkeypatch.setattr(ModelTrainerCli, "utility", classmethod(fake_utility))

    ModelTrainerCli._main(
        n_chars=100,
        overwrite=True,
        train_input_dir_path="{oot3d_data_path}/train_{n_chars}",
        test_input_dir_path="{oot3d_data_path}/test_{n_chars}",
        model_output_path="{oot3d_data_path}/model_{n_chars}.pth",
    )

    assert captured["exist_ok"] is True
    assert "overwrite" not in captured["run_kwargs"]
    assert "n_chars" not in captured["run_kwargs"]
