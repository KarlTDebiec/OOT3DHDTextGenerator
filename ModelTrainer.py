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
from IPython import embed
import pathlib
from typing import Union

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
        self.cache_file = conf["cache"]
        self.verbosity = conf["verbosity"]

    def __call__(self):
        pass

        # Load cache
        if isfile(self.cache_file):
            if not access(self.cache_file, R_OK):
                raise ValueError()
            if self.verbosity >= 1:
                print(f"Loading cache from '{self.cache_file}'")
            self.load_cache()
        else:
            # Generate images
            n_chars = 100
            fonts = ["/System/Library/Fonts/STHeiti Light.ttc",
                     "/System/Library/Fonts/STHeiti Medium.ttc",
                     "/Library/Fonts/Songti.ttc"]
            sizes = [11, 12, 13]
            offsets = [-1, 0, 1]
            rotations = [-7, 0, 7]
            n_images = len(hanzi_chars[:n_chars]) * len(fonts) * len(sizes) \
                       * len(offsets) * len(offsets) * len(rotations)

            self.all_images = np.zeros((n_images, 16, 16), np.uint8)
            self.all_labels = np.zeros(n_images, str)
            i = 0
            for char in hanzi_chars[:n_chars]:
                for font in fonts:
                    for size in sizes:
                        for offset in product(offsets, offsets):
                            for rotation in rotations:
                                data = self.draw_image(char, font=font,
                                                       size=size,
                                                       offset=offset,
                                                       rotation=rotation)
                                print(f"{i:06d}", char, font, size, offset,
                                      rotation)
                                print(data)
                                self.all_images[i] = data
                                self.all_labels[i] = char
                                i += 1

        # Train

        # Evaluate

        # Save cache
        if self.verbosity >= 1:
            print(f"Saving cache to '{self.cache_file}'")
        self.save_cache()

        # Save model

    # endregion

    # region Properties

    @property
    def all_images(self) -> np.ndarray:
        if not hasattr(self, "_all_images"):
            raise ValueError()
        return self._all_images

    @all_images.setter
    def all_images(self, value: np.ndarray):
        self._all_images = value

    @property
    def all_labels(self) -> np.ndarray:
        if not hasattr(self, "_all_labels"):
            raise ValueError()
        return self._all_labels

    @all_labels.setter
    def all_labels(self, value: np.ndarray):
        self._all_labels = value

    @property
    def cache_file(self) -> Union[str, None]:
        if not hasattr(self, "_cache_file"):
            self._cache_file = None
        return self._cache_file

    @cache_file.setter
    def cache_file(self, value: str):
        value = expandvars(value)
        if isfile(value):
            if not (access(value, R_OK) and access(value, W_OK)):
                raise ValueError()
        elif isdir(dirname(value)):
            if not (access(dirname(value), R_OK)
                    and access(dirname(value), W_OK)):
                raise ValueError()
        else:
            raise ValueError
        self._cache_file = value

    @property
    def trn_images(self) -> np.ndarray:
        if not hasattr(self, "_trn_images"):
            raise ValueError()
        return self._trn_images

    @trn_images.setter
    def trn_images(self, value: np.ndarray):
        self._trn_images = value

    @property
    def trn_labels(self) -> np.ndarray:
        if not hasattr(self, "_trn_labels"):
            raise ValueError()
        return self._trn_labels

    @trn_labels.setter
    def trn_labels(self, value: np.ndarray):
        self._trn_labels = value

    @property
    def tst_images(self) -> np.ndarray:
        if not hasattr(self, "_tst_images"):
            raise ValueError()
        return self._tst_images

    @tst_images.setter
    def tst_images(self, value: np.ndarray):
        self._tst_images = value

    @property
    def val_images(self) -> np.ndarray:
        if not hasattr(self, "_val_images"):
            raise ValueError()
        return self._val_images

    @val_images.setter
    def val_images(self, value: np.ndarray):
        self._val_images = value

    @property
    def tst_labels(self) -> np.ndarray:
        if not hasattr(self, "_tst_labels"):
            raise ValueError()
        return self.tst_labels

    @tst_labels.setter
    def tst_labels(self, value: np.ndarray):
        self.tst_labels = value

    @property
    def val_labels(self) -> np.ndarray:
        if not hasattr(self, "_val_labels"):
            raise ValueError()
        return self._val_labels

    @val_labels.setter
    def val_labels(self, value: np.ndarray):
        self._val_labels = value

    # endregion

    # region Methods

    def load_cache(self) -> None:

        with h5py.File(self.cache_file) as cache:
            # Load images
            if "images" in cache:
                self.all_images = np.array(cache["images/images"])
                self.all_labels = np.array(
                    [l.decode("UTF8") for l in
                     np.array(cache["images/labels"])])

    def save_cache(self) -> None:

        with h5py.File(self.cache_file) as cache:
            # Save images
            if "images" in cache:
                del cache["images"]
            cache.create_dataset("images/images",
                                 data=self.all_images,
                                 dtype=np.uint8,
                                 chunks=True,
                                 compression="gzip")
            cache.create_dataset("images/labels",
                                 data=[l.encode("UTF8") for l in
                                       self.all_labels],
                                 dtype="S4",
                                 chunks=True,
                                 compression="gzip")

    # endregion

    # region Static Methods

    @staticmethod
    def draw_image(char: str,
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
