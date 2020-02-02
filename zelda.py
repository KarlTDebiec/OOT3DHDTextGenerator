#!/usr/bin/python

from os import R_OK, access, listdir, W_OK
from os.path import expandvars, isdir, isfile, basename, dirname

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError
from IPython import embed
from collections import OrderedDict

import h5py


class ZeldaOOT3DHDTextGenerator(object):

    # region Builtins

    def __init__(self, conf_file: str = "conf.yaml"):

        # Read configuration
        if not (isfile(conf_file) and access(conf_file, R_OK)):
            raise ValueError()
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        self.dump_directory = conf["Dump"]
        self.cache_file = conf["Cache"]

    def __call__(self):

        # Load cache
        self.load_cache()

        # Review all existing images
        for i, filename in enumerate(listdir(self.dump_directory)):

            # If filename is known, check if confirmed or unconfirmed
            if filename in self.confirmed_texts:
                continue
            elif filename in self.unconfirmed_texts:
                if self.is_text_confirmable(filename):
                    self.confirm_text(filename)
                continue

            # If file is not a text image, skip
            if not self.is_file_text_image(filename):
                continue

            # Otherwise, add new text
            self.add_text(filename)

        # Assign unassigned characters
        self.assign_chars()

        # Save cache
        self.save_cache()

    # endregion

    # region Properties

    @property
    def cache_file(self) -> str:
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
    def chars(self) -> OrderedDict:
        if not hasattr(self, "_chars"):
            self._chars = OrderedDict()
        return self._chars

    @chars.setter
    def chars(self, value: OrderedDict):
        if not (isinstance(value, OrderedDict)):
            raise ValueError()
        self._chars = value

    @property
    def char_bytes(self) -> list:
        return list(self.chars.keys())

    @property
    def char_confirmations(self) -> list:
        return [c[1] for c in self.chars.values()]

    @property
    def char_assignments(self) -> list:
        return [c[0] for c in self.chars.values()]

    @property
    def char_images(self) -> np.ndarray:
        return np.stack([np.frombuffer(k, dtype=np.uint8)
                         for k in self.char_bytes]).reshape((-1, 16, 16))

    @property
    def confirmed_texts(self) -> dict:
        if not hasattr(self, "_confirmed_texts"):
            self._confirmed_texts = {}
        return self._confirmed_texts

    @confirmed_texts.setter
    def confirmed_texts(self, value: dict):
        if not (isinstance(value, dict)):
            raise ValueError()
        self._confirmed_texts = value

    @property
    def dump_directory(self) -> str:
        if not hasattr(self, "_dump_directory"):
            raise ValueError()
        return self._dump_directory

    @dump_directory.setter
    def dump_directory(self, value: str):
        value = expandvars(value)
        if not (isdir(value) and access(value, R_OK)):
            raise ValueError()
        self._dump_directory = value

    @property
    def unconfirmed_texts(self) -> dict:
        if not hasattr(self, "_unconfirmed_texts"):
            self._unconfirmed_texts = {}
        return self._unconfirmed_texts

    @unconfirmed_texts.setter
    def unconfirmed_texts(self, value: dict):
        if not (isinstance(value, dict)):
            raise ValueError()
        self._unconfirmed_texts = value

    @property
    def verbosity(self) -> int:
        """int: Level of output to provide"""
        if not hasattr(self, "_verbosity"):
            self._verbosity = 1
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: int):
        if not isinstance(value, int) and value >= 0:
            raise ValueError()
        self._verbosity = value

    # endregion

    # region Methods

    def add_text(self, filename: str) -> None:
        if filename in self.confirmed_texts:
            raise ValueError()
        elif filename in self.unconfirmed_texts:
            raise ValueError()
        text_data = np.array(Image.open(f"{self.dump_directory}/{filename}"))
        text_data = text_data[:, :, 3]
        indexes = []
        for x in range(16):
            for y in range(16):
                char_data = text_data[x * 16:(x + 1) * 16, y * 16:(y + 1) * 16]
                char_bytes = char_data.tobytes()
                if char_bytes not in self.chars:
                    self.chars[char_bytes] = ("", False)
                indexes.append(self.char_bytes.index(char_bytes))
        self.unconfirmed_texts[filename] = np.array(indexes, np.uint32)
        if self.is_text_confirmable(filename):
            self.confirm_text(filename)

    def assign_chars(self) -> None:
        for char, (assignment, confirmed) in self.chars.items():
            if not confirmed:
                data = np.frombuffer(char, dtype=np.uint8).reshape(16, 16)
                print(data)
                image = Image.fromarray(data).resize((160, 160), Image.NEAREST)
                self.show_image(image)
                try:
                    assignment = input("Assignment:")
                    if assignment != "":
                        self.chars[char] = (assignment, True)
                except KeyboardInterrupt:
                    break

    def confirm_text(self, filename: str) -> None:
        if filename in self.confirmed_texts:
            raise ValueError()
        elif filename not in self.unconfirmed_texts:
            raise ValueError()

        self.confirmed_texts[filename] = self.get_text(filename)
        del self.unconfirmed_texts[filename]

    def get_text(self, filename: str) -> str:
        if filename not in self.unconfirmed_texts:
            raise ValueError()

        indexes = self.unconfirmed_texts[filename]
        assignments = [self.char_assignments[i] for i in indexes]
        return "".join(assignments).rstrip()

    def is_file_text_image(self, filename: str) -> bool:
        try:
            data = np.array(Image.open(f"{self.dump_directory}/{filename}"))
        except UnidentifiedImageError:
            return False
        if data.shape != (256, 256, 4):
            return False
        if data[:, :, :3].sum() != 0:
            return False
        return True

    def is_text_confirmable(self, filename: str) -> bool:
        if filename not in self.unconfirmed_texts:
            raise ValueError()

        indexes = self.unconfirmed_texts[filename]
        confirmations = [self.char_confirmations[i] for i in indexes]
        if np.all(confirmations):
            return True
        else:
            return False

    def load_cache(self) -> None:

        with h5py.File(self.cache_file) as cache:

            # Load characters
            if "characters" in cache:
                assignments = [a.decode("UTF8") for a in
                               np.array(cache["characters/assignments"])]
                confirmations = np.array(cache["characters/confirmations"])
                images = np.array(cache["characters/images"])
                for i, a, c in zip(images, assignments, confirmations):
                    self.chars[i.tobytes()] = (a, c)

            # Load unconfirmed texts
            if "texts/unconfirmed" in cache:
                filenames = [f.decode("UTF8") for f in
                             np.array(cache["texts/unconfirmed/filenames"])]
                indexes = np.array(cache["texts/unconfirmed/indexes"])
                for cache, i in zip(filenames, indexes):
                    self.unconfirmed_texts[cache] = i

            # Load confirmed texts
            if "texts/confirmed" in cache:
                filenames = [f.decode("UTF8") for f in
                             np.array(cache["texts/confirmed/filenames"])]
                texts = [f.decode("UTF8") for f in
                         np.array(cache["texts/confirmed/text"])]
                for f, t in zip(filenames, texts):
                    self.confirmed_texts[f] = t

    def save_cache(self) -> None:

        with h5py.File(self.cache_file) as cache:

            # Save characters
            if "characters" in cache:
                del cache["characters"]
            cache.create_dataset("characters/assignments",
                                 data=[a.encode("UTF8") for a in
                                       self.char_assignments],
                                 dtype="S4",
                                 chunks=True,
                                 compression="gzip")
            cache.create_dataset("characters/confirmations",
                                 data=self.char_confirmations,
                                 dtype=bool,
                                 chunks=True,
                                 compression="gzip")
            cache.create_dataset("characters/images",
                                 data=self.char_images,
                                 dtype=np.uint8,
                                 chunks=True,
                                 compression="gzip")

            # Save unconfirmed texts
            if "texts/unconfirmed" in cache:
                del cache["texts/unconfirmed"]
            if len(self.unconfirmed_texts) > 0:
                cache.create_dataset("texts/unconfirmed/filenames",
                                     data=[k.encode("UTF8") for k in
                                           self.unconfirmed_texts.keys()],
                                     dtype="S48",
                                     chunks=True,
                                     compression="gzip")
                cache.create_dataset("texts/unconfirmed/indexes",
                                     data=np.stack(
                                         list(
                                             self.unconfirmed_texts.values())),
                                     dtype=np.uint32,
                                     chunks=True,
                                     compression="gzip")

            # Save confirmed texts
            if "texts/confirmed" in cache:
                del cache["texts/confirmed"]
            if len(self.confirmed_texts) > 0:
                cache.create_dataset("texts/confirmed/filenames",
                                     data=[k.encode("UTF8") for k in
                                           self.confirmed_texts.keys()],
                                     dtype="S48",
                                     chunks=True,
                                     compression="gzip")
                cache.create_dataset("texts/confirmed/text",
                                     data=[t.encode("UTF8") for t in
                                           self.confirmed_texts.values()],
                                     dtype="S1024",
                                     chunks=True,
                                     compression="gzip")

    # Static Methods

    @staticmethod
    def show_image(image: Image.Image) -> None:
        try:
            from imgcat import imgcat

            imgcat(image)
        except ImportError:
            image.show()

    # endregion


if __name__ == "__main__":
    ZeldaOOT3DHDTextGenerator()()
