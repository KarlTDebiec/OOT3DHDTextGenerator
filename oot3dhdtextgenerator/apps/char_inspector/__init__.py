#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character inspector Flask application.

This module may import from: common, core, data, image, utilities

Hierarchy within module:
* char_inspector
"""

from __future__ import annotations

from oot3dhdtextgenerator.apps.char_inspector.char_inspector import CharInspector

__all__ = [
    "CharInspector",
]
