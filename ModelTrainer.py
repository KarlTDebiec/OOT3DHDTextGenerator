#!/usr/bin/python
# -*- coding: utf-8 -*-
#   ModelTrainer.py
#
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license.
################################### MODULES ###################################
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from os import R_OK, W_OK, access, listdir
from os.path import dirname, expandvars, isdir, isfile, basename
import h5py
from itertools import product
import yaml
from sys import modules
from IPython import embed
import pathlib

################################## VARIABLES ##################################
package_root = str(pathlib.Path(__file__).parent.absolute())
hanzi_frequency = pd.read_csv(f"{package_root}/data/characters.txt",
                              sep="\t", names=["character", "frequency",
                                               "cumulative frequency"])
hanzi_chars = np.array(hanzi_frequency["character"], np.str)


################################### CLASSES ###################################
class ModelTrainer():

    # region Builtins

    def __init__(self, conf_file: str = "conf_train.yaml"):
        # Read configuration
        if not (isfile(conf_file) and access(conf_file, R_OK)):
            raise ValueError()
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        self.verbosity = conf["verbosity"]

    def __call__(self):
        pass

        # Load cache

        # Load model

        # Generate images
        fonts = ["/System/Library/Fonts/STHeiti Light.ttc",
                 "/System/Library/Fonts/STHeiti Medium.ttc",
                 "/Library/Fonts/Songti.ttc"]
        sizes = [11, 12, 13]
        offsets = [-1, 0, 1]
        rotations = [-7, 0, 7]
        n_images = len(hanzi_chars[:10]) * len(fonts) * len(sizes) \
                   * len(offsets) * len(offsets) * len(rotations)
        images = np.zeros((n_images, 16, 16), np.uint8)
        labels = np.zeros(n_images, str)
        i = 0
        for char in hanzi_chars[:10]:
            for font in fonts:
                for size in sizes:
                    for offset in product(offsets, offsets):
                        for rotation in rotations:
                            data = self.draw_image(char, font=font, size=size,
                                                   offset=offset,
                                                   rotation=rotation)
                            print(f"{i:06d}", char, font, size, offset,
                                  rotation)
                            images[i] = data
                            labels[i] = char
                            i += 1
        embed()

        # Train

        # Evaluate

        # Save cache

        # Save model

        # endregion

        # region Properties

    @property
    def font(self) -> str:
        if not hasattr(self, "_font"):
            raise ValueError()
        return self._font

    @font.setter
    def font(self, value: str):
        self._font = value

    @property
    def fontsize(self) -> int:
        if not hasattr(self, "_fontsize"):
            raise ValueError()
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value: int):
        self._fontsize = value

    # endregion

    # region Methods

    def draw_image(self, char: str,
                   font: str = "/System/Library/Fonts/STHeiti Light.ttc",
                   size: int = 12, offset: tuple = (0, 0),
                   rotation: int = 0):
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font, size)
        width, height = draw.textsize(char, font=font)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font, fill=255)
        image = image.rotate(rotation)
        data = np.array(image)
        data = np.roll(data, offset, (0, 1))

        return data

    # endregion


#################################### MAIN #####################################
if __name__ == "__main__":
    ModelTrainer()()
