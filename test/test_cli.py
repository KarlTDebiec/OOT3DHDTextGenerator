#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for oot3dhdtextgenerator command-line interface."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from inspect import getfile
from io import StringIO
from pathlib import Path
from typing import Any

from pipescaler.testing.mark import parametrize_with_readable_ids
from pytest import MonkeyPatch

import oot3dhdtextgenerator.cli.model_trainer_cli as model_trainer_cli_module
from oot3dhdtextgenerator.cli import (
    CharAssignerCli,
    CharInspectorCli,
    ModelTrainerCli,
    TrainingDatasetGeneratorCli,
)
from oot3dhdtextgenerator.cli import (
    char_assigner_cli as char_assigner_cli_module,
)
from oot3dhdtextgenerator.cli import (
    char_inspector_cli as char_inspector_cli_module,
)
from oot3dhdtextgenerator.cli import (
    training_dataset_generator_cli as training_dataset_generator_cli_module,
)
from oot3dhdtextgenerator.common import CommandLineInterface
from oot3dhdtextgenerator.common.testing import run_cli_with_args


@parametrize_with_readable_ids(
    ("command"),
    [
        (CharAssignerCli),
        (CharInspectorCli),
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


def test_training_dataset_generator_overwrite_arg() -> None:
    """Test training dataset generator overwrite argument parsing."""
    parser = TrainingDatasetGeneratorCli.argparser()
    assert parser.parse_args([]).overwrite is False
    assert parser.parse_args(["-o"]).overwrite is True
    assert parser.parse_args(["--overwrite"]).overwrite is True


def test_training_dataset_generator_overwrite_uses_exist_ok(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test training dataset overwrite option is forwarded to output validation."""
    captured: dict[str, Any] = {}

    def fake_val_output_dir_path(value: str) -> Path:
        """Validate mocked output directory path."""
        return Path(value)

    def fake_val_output_path(value: Path, exist_ok: bool = False) -> Path:
        """Validate mocked output path."""
        captured.setdefault("exist_ok", []).append(exist_ok)
        return value

    class FakeUtility:
        """Mock utility class."""

        @staticmethod
        def run(**kwargs: Any):
            """Capture run keyword arguments."""
            captured["run_kwargs"] = kwargs

    def fake_utility(
        cls: type[TrainingDatasetGeneratorCli],
    ) -> type[FakeUtility]:
        """Get mocked utility."""
        _ = cls
        return FakeUtility

    monkeypatch.setattr(
        training_dataset_generator_cli_module,
        "val_output_dir_path",
        fake_val_output_dir_path,
    )
    monkeypatch.setattr(
        training_dataset_generator_cli_module, "val_output_path", fake_val_output_path
    )
    monkeypatch.setattr(
        TrainingDatasetGeneratorCli, "utility", classmethod(fake_utility)
    )

    TrainingDatasetGeneratorCli._main(
        n_chars=100,
        overwrite=True,
        test_proportion=0.1,
        train_output_dir_path="{oot3d_data_path}/train_{n_chars}",
        test_output_dir_path="{oot3d_data_path}/test_{n_chars}",
    )

    assert captured["exist_ok"] == [True, True, True, True]
    assert "overwrite" not in captured["run_kwargs"]


def test_char_assigner_default_paths_are_formatted(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test char assigner defaults are formatted with oot3d data path."""
    captured: dict[str, Any] = {}

    def fake_val_output_dir_path(value: str) -> Path:
        """Validate mocked output directory path."""
        captured["assignment_dir_path"] = value
        return Path(value)

    def fake_val_input_path(value: str) -> Path:
        """Validate mocked input path."""
        captured["model_input_path"] = value
        return Path(value)

    class FakeCharAssigner:
        """Mock character assigner."""

        def __init__(self, **kwargs: Any) -> None:
            """Capture initializer keyword arguments."""
            captured["init_kwargs"] = kwargs

        def run(self, **kwargs: Any) -> None:
            """Capture run keyword arguments."""
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        char_assigner_cli_module, "val_output_dir_path", fake_val_output_dir_path
    )
    monkeypatch.setattr(char_assigner_cli_module, "val_input_path", fake_val_input_path)
    monkeypatch.setattr(char_assigner_cli_module, "CharAssigner", FakeCharAssigner)

    CharAssignerCli._main(
        n_chars=100,
        assignment_dir_path="{oot3d_data_path}",
        model_input_path="{oot3d_data_path}/model_{n_chars}.pth",
        cuda_enabled=True,
        mps_enabled=True,
    )

    assert captured["assignment_dir_path"].endswith("/oot3d")
    assert captured["model_input_path"].endswith("/oot3d/model_100.pth")
    assert captured["run_kwargs"] == {"port": 5001}


def test_char_assigner_custom_port_is_forwarded(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test char assigner forwards custom port to Flask app run."""
    captured: dict[str, Any] = {}

    def fake_val_output_dir_path(value: str) -> Path:
        """Validate mocked output directory path."""
        return Path(value)

    def fake_val_input_path(value: str) -> Path:
        """Validate mocked input path."""
        return Path(value)

    class FakeCharAssigner:
        """Mock character assigner."""

        def __init__(self, **kwargs: Any) -> None:
            """Capture initializer keyword arguments."""
            captured["init_kwargs"] = kwargs

        def run(self, **kwargs: Any) -> None:
            """Capture run keyword arguments."""
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        char_assigner_cli_module, "val_output_dir_path", fake_val_output_dir_path
    )
    monkeypatch.setattr(char_assigner_cli_module, "val_input_path", fake_val_input_path)
    monkeypatch.setattr(char_assigner_cli_module, "CharAssigner", FakeCharAssigner)

    CharAssignerCli._main(
        n_chars=100,
        assignment_dir_path="{oot3d_data_path}",
        model_input_path="{oot3d_data_path}/model_{n_chars}.pth",
        cuda_enabled=True,
        mps_enabled=True,
        port=5002,
    )

    assert captured["run_kwargs"] == {"port": 5002}
    assert "port" not in captured["init_kwargs"]


def test_char_inspector_default_paths_are_formatted(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test char inspector defaults are formatted with oot3d data path."""
    captured: dict[str, Any] = {}

    def fake_val_output_dir_path(value: str) -> Path:
        """Validate mocked output directory path."""
        captured["assignment_dir_path"] = value
        return Path(value)

    class FakeCharInspector:
        """Mock character inspector."""

        def __init__(self, **kwargs: Any) -> None:
            """Capture initializer keyword arguments."""
            captured["init_kwargs"] = kwargs

        def run(self, **kwargs: Any) -> None:
            """Capture run keyword arguments."""
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        char_inspector_cli_module, "val_output_dir_path", fake_val_output_dir_path
    )
    monkeypatch.setattr(char_inspector_cli_module, "CharInspector", FakeCharInspector)

    CharInspectorCli._main(
        n_chars=100,
        assignment_dir_path="{oot3d_data_path}",
    )

    assert captured["assignment_dir_path"].endswith("/oot3d")
    assert captured["run_kwargs"] == {"port": 5002}


def test_char_inspector_custom_port_is_forwarded(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test char inspector forwards custom port to Flask app run."""
    captured: dict[str, Any] = {}

    def fake_val_output_dir_path(value: str) -> Path:
        """Validate mocked output directory path."""
        return Path(value)

    class FakeCharInspector:
        """Mock character inspector."""

        def __init__(self, **kwargs: Any) -> None:
            """Capture initializer keyword arguments."""
            captured["init_kwargs"] = kwargs

        def run(self, **kwargs: Any) -> None:
            """Capture run keyword arguments."""
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        char_inspector_cli_module, "val_output_dir_path", fake_val_output_dir_path
    )
    monkeypatch.setattr(char_inspector_cli_module, "CharInspector", FakeCharInspector)

    CharInspectorCli._main(
        n_chars=100,
        assignment_dir_path="{oot3d_data_path}",
        port=5003,
    )

    assert captured["run_kwargs"] == {"port": 5003}
    assert "port" not in captured["init_kwargs"]
