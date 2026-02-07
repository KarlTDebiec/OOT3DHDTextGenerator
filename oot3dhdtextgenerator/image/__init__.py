#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Code related to images."""

from __future__ import annotations

from .oot3d_shadow_processor import OOT3DShadowProcessor
from .oot3d_text_processor import OOT3DHDTextProcessor

__all__ = [
    "OOT3DHDTextProcessor",
    "OOT3DShadowProcessor",
]
