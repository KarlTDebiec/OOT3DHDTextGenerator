#!/usr/bin/env python
#   ModelTrainer.py
#
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license.
################################### MODULES ###################################
import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from itertools import product
from os import R_OK, W_OK, access
from os.path import dirname, expandvars, isdir, isfile
from pathlib import Path
from tensorflow import keras
from typing import Optional

################################## VARIABLES ##################################
package_root = str(Path(__file__).parent.absolute())
hanzi_frequency = pd.read_csv(
    f"{package_root}/data/characters.txt",
    sep="\t", names=["character", "frequency", "cumulative frequency"])
hanzi_chars = np.array(hanzi_frequency["character"], np.str)
n_chars = 9933


################################### CLASSES ###################################
class ModelTrainer:

    # region Builtins

    def __init__(self, conf_file: str = "conf_train.yaml"):
        # Read configuration
        if not (isfile(conf_file) and access(conf_file, R_OK)):
            raise ValueError()
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        self.cache_file = conf["cache"]
        self.verbosity = conf["verbosity"]

    def __call__(self) -> None:
        # Load cache
        if isfile(self.cache_file):
            if not access(self.cache_file, R_OK):
                raise ValueError()
            if self.verbosity >= 1:
                print(f"Loading cache from '{self.cache_file}'")
            self.load_cache()
        else:
            # Generate images
            self.draw_images()

        # Save cache
        if self.verbosity >= 1:
            print(f"Saving cache to '{self.cache_file}'")
        self.save_cache()

        # Train
        self.select_trn_tst_val_images()

        epochs = 100
        batch_size = n_chars
        sorter = np.argsort(hanzi_chars)
        trn_images = np.expand_dims(
            self.trn_images.astype(np.float16) / 255.0, axis=3)
        trn_labels = np.array(sorter[np.searchsorted(
            hanzi_chars, self.trn_labels, sorter=sorter)])
        tst_images = np.expand_dims(
            self.tst_images.astype(np.float16) / 255.0, axis=3)
        tst_labels = np.array(sorter[np.searchsorted(
            hanzi_chars, self.tst_labels, sorter=sorter)])
        val_images = np.expand_dims(
            self.val_images.astype(np.float16) / 255.0, axis=3)
        val_labels = np.array(sorter[np.searchsorted(
            hanzi_chars, self.val_labels, sorter=sorter)])

        model = self.get_model()
        print(model.summary())
        model.fit(trn_images, trn_labels, epochs=epochs, batch_size=batch_size,
                  validation_data=(val_images, val_labels),
                  callbacks=[
                      keras.callbacks.EarlyStopping(
                          monitor="val_loss", min_delta=0.01, patience=3,
                          verbose=1),
                      keras.callbacks.ModelCheckpoint(
                          f"{dirname(str(self.cache_file))}/{n_chars:05d}_"
                          "{epoch:03d}_{val_accuracy:6.4f}.h5",
                          monitor="val_accuracy", verbose=1)])

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
    def all_images(self, value: np.ndarray) -> None:
        self._all_images = value

    @property
    def all_labels(self) -> np.ndarray:
        if not hasattr(self, "_all_labels"):
            raise ValueError()
        return self._all_labels

    @all_labels.setter
    def all_labels(self, value: np.ndarray) -> None:
        self._all_labels = value

    @property
    def cache_file(self) -> Optional[str]:
        if not hasattr(self, "_cache_file"):
            self._cache_file: Optional[str] = None
        return self._cache_file

    @cache_file.setter
    def cache_file(self, value: Optional[str]) -> None:
        if value is not None:
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
    def trn_images(self, value: np.ndarray) -> None:
        self._trn_images = value

    @property
    def trn_labels(self) -> np.ndarray:
        if not hasattr(self, "_trn_labels"):
            raise ValueError()
        return self._trn_labels

    @trn_labels.setter
    def trn_labels(self, value: np.ndarray) -> None:
        self._trn_labels = value

    @property
    def tst_images(self) -> np.ndarray:
        if not hasattr(self, "_tst_images"):
            raise ValueError()
        return self._tst_images

    @tst_images.setter
    def tst_images(self, value: np.ndarray) -> None:
        self._tst_images = value

    @property
    def tst_labels(self) -> np.ndarray:
        if not hasattr(self, "_tst_labels"):
            raise ValueError()
        return self._tst_labels

    @tst_labels.setter
    def tst_labels(self, value: np.ndarray) -> None:
        self._tst_labels = value

    @property
    def val_images(self) -> np.ndarray:
        if not hasattr(self, "_val_images"):
            raise ValueError()
        return self._val_images

    @val_images.setter
    def val_images(self, value: np.ndarray) -> None:
        self._val_images = value

    @property
    def val_labels(self) -> np.ndarray:
        if not hasattr(self, "_val_labels"):
            raise ValueError()
        return self._val_labels

    @val_labels.setter
    def val_labels(self, value: np.ndarray) -> None:
        self._val_labels = value

    # endregion

    # region Methods

    def draw_images(self) -> None:

        fonts = ["/System/Library/Fonts/STHeiti Light.ttc",
                 "/System/Library/Fonts/STHeiti Medium.ttc",
                 "/Library/Fonts/Songti.ttc"]
        sizes = [15, 16]
        offsets = [-1, 0, 1]
        fills = [215, 225, 235, 245, 255]
        rotations = [0]
        n_images = len(hanzi_chars[:n_chars]) * len(fonts) * len(sizes) \
                   * len(fills) * len(offsets) * len(offsets) \
                   * len(rotations)
        self.all_images = np.zeros((n_images, 16, 16), np.uint8)
        self.all_labels = np.zeros(n_images, str)
        i = 0
        for j, char in enumerate(hanzi_chars[:n_chars]):
            print(j, char)
            for font in fonts:
                for size in sizes:
                    for fill in fills:
                        for offset in product(offsets, offsets):
                            for rotation in rotations:
                                data = self.draw_image(
                                    char, font=font, size=size,
                                    fill=fill, offset=offset,
                                    rotation=rotation)
                                # print(f"{i:08d}", char, font, size,
                                #       fill, offset, rotation)
                                # print(data)
                                self.all_images[i] = data
                                self.all_labels[i] = char
                                i += 1

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

    def select_trn_tst_val_images(self, tst_portion: float = 0.1,
                                  val_portion: float = 0.1) -> None:
        from random import sample

        # Select indexes
        trn_indexes = set(range(self.all_labels.size))
        tst_indexes = set(sample(
            trn_indexes, int(np.floor(self.all_labels.size * tst_portion))))
        trn_indexes -= tst_indexes
        val_indexes = set(sample(
            trn_indexes, int(np.floor(self.all_labels.size * val_portion))))
        trn_indexes -= val_indexes
        trn_indexes = list(trn_indexes)
        tst_indexes = list(tst_indexes)
        val_indexes = list(val_indexes)

        # Store
        self.trn_images = self.all_images[trn_indexes]
        self.trn_labels = self.all_labels[trn_indexes]
        self.tst_images = self.all_images[tst_indexes]
        self.tst_labels = self.all_labels[tst_indexes]
        self.val_images = self.all_images[val_indexes]
        self.val_labels = self.all_labels[val_indexes]

    # endregion

    # region Static Methods

    @staticmethod
    def draw_image(char: str,
                   font: str = "/System/Library/Fonts/STHeiti Light.ttc",
                   size: int = 12,
                   fill: int = 0,
                   offset: tuple = (0, 0),
                   rotation: int = 0) -> np.ndarray:
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font, size)
        width, height = draw.textsize(char, font=font)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font, fill=fill)
        image = image.rotate(rotation)
        data = np.array(image)
        data = np.roll(data, offset, (0, 1))

        return data

    @staticmethod
    def get_model() -> keras.Sequential:
        model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=2, padding="same",
                                activation="relu", input_shape=(16, 16, 1)),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(filters=128, kernel_size=2, padding="same",
                                activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(filters=256, kernel_size=2, padding="same",
                                activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(n_chars, activation="softmax")
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    # endregion


#################################### MAIN #####################################
if __name__ == "__main__":
    ModelTrainer()()
