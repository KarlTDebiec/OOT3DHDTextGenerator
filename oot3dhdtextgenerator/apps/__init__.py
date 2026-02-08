#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Flask applications.

This module may import from: common, core, data, image, utilities

Hierarchy within module:
* char_assigner
"""

from __future__ import annotations

from oot3dhdtextgenerator.apps.char_assigner.char_assigner import CharAssigner

__all__ = [
    "CharAssigner",
]
