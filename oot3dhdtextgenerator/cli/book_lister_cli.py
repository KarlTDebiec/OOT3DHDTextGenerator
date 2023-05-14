#!/usr/bin/env python
#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner command-line interface."""
from __future__ import annotations

from typing import Any

from oot3dhdtextgenerator.apps.book_lister import BookLister
from oot3dhdtextgenerator.common import CommandLineInterface


class BookListerCli(CommandLineInterface):
    """Character assigner command-line interface."""

    @classmethod
    def main_internal(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments."""
        char_assigner = BookLister(**kwargs)
        char_assigner.run(port=5001)


if __name__ == "__main__":
    BookListerCli.main()
