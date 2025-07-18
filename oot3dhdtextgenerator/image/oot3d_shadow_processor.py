#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Processes shadow images."""

from __future__ import annotations

import numpy as np
from PIL import Image
from pipescaler.image.operators.processors import PotraceProcessor

from oot3dhdtextgenerator.image.typing import RGBA


class OOT3DShadowProcessor(PotraceProcessor):
    """Processes shadow images."""

    def __init__(
        self,
        arguments: str = "-b svg -k 0.3 -a 1.34 -O 0.2",
        invert: bool = False,
        scale: float = 4.0,
    ) -> None:
        """Initialize.

        Arguments:
            arguments: Command-line arguments to pass to potrace
            invert: Whether to invert image before tracing
            scale: Scale of re-rasterized output image relative to input
        """
        super().__init__(arguments=arguments, invert=invert, scale=scale)

    def __call__(self, input_image: Image.Image) -> Image.Image:
        """Process shadow image and return traced output."""
        # Flatten image and convert to monochrome
        canvas = Image.new(RGBA, input_image.size, (255, 255, 255))
        composite = Image.alpha_composite(canvas, input_image)
        monochrome_image = composite.point(lambda p: p > 240 and 255)

        # Trace image using potrace
        traced_image = super().__call__(monochrome_image).convert("L")

        # Convert back to shadow
        output_data = np.zeros((traced_image.height, traced_image.width, 4))
        output_data[:, :, 3] = (255 - np.array(traced_image)) * 0.666667
        output_image = Image.fromarray(output_data.astype(np.uint8))

        return output_image

    @classmethod
    def help_markdown(cls) -> str:
        """Short description of this tool in markdown, with links."""
        return (
            "Traces OOT 3D shadow image using "
            "[Potrace](http://potrace.sourceforge.net/) and re-rasterizes, optionally "
            "resizing."
        )

    @classmethod
    def inputs(cls) -> dict[str, tuple[str, ...]]:
        """Inputs to this operator."""
        return {
            "input": (RGBA,),
        }

    @classmethod
    def outputs(cls) -> dict[str, tuple[str, ...]]:
        """Outputs of this operator."""
        return {
            "output": (RGBA,),
        }
