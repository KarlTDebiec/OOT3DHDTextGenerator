#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Type definitions for image utilities."""

from __future__ import annotations

from typing import Literal

ImageMode = Literal["RGBA"]
"""Supported image mode."""

RGBA: ImageMode = "RGBA"
"""RGBA image mode constant."""
