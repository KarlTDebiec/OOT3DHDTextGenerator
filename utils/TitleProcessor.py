#!python
# -*- coding: utf-8 -*-
#   TitleProcessor.py
#
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license.
################################### MODULES ###################################
from __future__ import annotations

from sys import argv

import numba as nb
import numpy as np
from PIL import Image


################################## FUNCTIONS ##################################
@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def denoise(paletted_data):
    for x in range(1, paletted_data.shape[1] - 1):
        for y in range(1, paletted_data.shape[0] - 1):
            slc = paletted_data[y - 1:y + 2, x - 1:x + 2]
            n_transparent = (slc == 127).sum()
            n_black = (slc == 0).sum()
            n_white = (slc == 255).sum()
            if paletted_data[y, x] == 127:
                if n_transparent < 4:
                    if n_black > n_white:
                        paletted_data[y, x] = 0
                    else:
                        paletted_data[y, x] = 255
            elif paletted_data[y, x] == 0:
                if n_black < 4:
                    if n_transparent > n_white:
                        paletted_data[y, x] = 127
                    else:
                        paletted_data[y, x] = 255
            elif paletted_data[y, x] == 255:
                if n_white < 4:
                    if n_transparent > n_black:
                        paletted_data[y, x] = 127
                    else:
                        paletted_data[y, x] = 0


#################################### MAIN #####################################


if __name__ == "__main__":
    # Parse arguments
    infile = argv[1]
    outfile = argv[2]

    # Load image data
    raw_image = Image.open(infile)
    raw_data = np.array(raw_image)
    grayscale_data = np.mean(raw_data[:, :, :2], axis=2).astype(np.uint8)
    alpha_data = raw_data[:, :, 3]

    # Restrict to palette of transparent, black, and white
    paletted_data = np.zeros_like(grayscale_data)  # Start all transparent
    paletted_data[alpha_data < 128] = 127
    paletted_data[
        (alpha_data > 128) & (grayscale_data > 128)] = 255  # Set white

    # Denoise
    denoise(paletted_data)

    # Reconstruct RGBA image from palette
    processed_data = np.zeros_like(raw_data)  # Start all black transparent
    processed_data[:, :, 3][paletted_data != 127] = 255
    processed_data[:, :, :3][paletted_data == 255] = 255
    processed_image = Image.fromarray(processed_data)

    # Save
    scaled_image = processed_image.resize((1024, 128), Image.BICUBIC)
    scaled_image.save(outfile)
