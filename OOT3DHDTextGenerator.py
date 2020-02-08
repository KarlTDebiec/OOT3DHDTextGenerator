#!/usr/bin/python
# -*- coding: utf-8 -*-
#   OOT3DHDTextGenerator.py
#
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license.
################################### MODULES ###################################
from collections import OrderedDict
from os import R_OK, W_OK, access, listdir
from os.path import basename, dirname, expandvars, isdir, isfile
from pathlib import Path
from readline import insert_text, redisplay, set_pre_input_hook
from shutil import copyfile
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from tensorflow import keras
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

################################## VARIABLES ##################################
package_root = str(Path(__file__).parent.absolute())
hanzi_frequency = pd.read_csv(
    f"{package_root}/data/characters.txt",
    sep="\t", names=["character", "frequency", "cumulative frequency"])
hanzi_chars = np.array(hanzi_frequency["character"], np.str)
n_chars = 9933


################################### CLASSES ###################################
class OOT3DHDTextGenerator():
    # TODO: Document

    # region Classes

    class FileCreatedEventHandler(FileSystemEventHandler):  # type: ignore

        def __init__(self, host) -> None:  # type: ignore
            self.host = host

        def on_created(self, event):  # type: ignore
            filename = basename(event.key[1])
            self.host.process_file(filename, True)

    # endregion

    # region Builtins

    def __init__(self, conf_file: str = "conf_cn.yaml"):

        # Read configuration
        if not (isfile(conf_file) and access(conf_file, R_OK)):
            raise ValueError()
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        self.backup_directory = conf["backup"]
        self.cache_file = conf["cache"]
        self.dump_directory = conf["dump"]
        self.font = conf["font"]
        self.fontsize = conf["fontsize"]
        self.language = conf["language"]
        self.load_directory = conf["load"]
        self.model_file = conf["model"]
        self.overwrite = conf["overwrite"]
        self.scale = conf["scale"]
        self.verbosity = conf["verbosity"]
        self.watch = conf["watch"]

    def __call__(self) -> None:

        # Load cache
        # TODO: If cache file does not exist, don't try to open it
        if self.verbosity >= 1:
            print(f"Loading cache from '{self.cache_file}'")
        self.load_cache()

        # Review all existing images
        if self.verbosity >= 1:
            print(f"Loading images from  '{self.dump_directory}'")
        for filename in listdir(self.dump_directory):
            self.process_file(filename)

        # Assign unassigned characters
        self.manually_assign_chars()

        # Save cache
        if self.verbosity >= 1:
            print(f"Saving cache to '{self.cache_file}'")
        self.save_cache()

        # Save text images
        if self.verbosity >= 1:
            print(f"Saving images to '{self.load_directory}'")
        for filename in self.confirmed_texts:
            if self.confirmed_texts[filename][0] == self.language:
                self.save_text(filename)

        # Watch for additional images and process as they appear
        if self.watch:
            if self.verbosity >= 1:
                print(f"Watching for new images in  '{self.dump_directory}'")
            self.observer.start()
            try:
                while True:
                    sleep(1)
            except KeyboardInterrupt:
                self.observer.stop()
            self.observer.join()

            # Resave cache after watching
            if self.verbosity >= 1:
                print(f"Saving cache to '{self.cache_file}'")
            self.save_cache()

    # endregion

    # region Properties

    @property
    def backup_directory(self) -> Optional[str]:
        if not hasattr(self, "_backup_directory"):
            self._backup_directory: Optional[str] = None
        return self._backup_directory

    @backup_directory.setter
    def backup_directory(self, value: Optional[str]) -> None:
        if value is not None:
            value = expandvars(value)
            # TODO: Create if possible
            if not (isdir(value) and access(value, W_OK)):
                raise ValueError()
        self._backup_directory = value

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
    def chars(self) -> OrderedDict[bytes, Tuple[str, bool]]:
        if not hasattr(self, "_chars"):
            self._chars: OrderedDict[bytes, Tuple[str, bool]] = OrderedDict()
        return self._chars

    @chars.setter
    def chars(self, value: OrderedDict[bytes, Tuple[str, bool]]) -> None:
        if not (isinstance(value, OrderedDict)):
            raise ValueError()
        self._chars = value

    @property
    def char_bytes(self) -> List[bytes]:
        return list(self.chars.keys())

    @property
    def char_confirmations(self) -> List[bool]:
        return [c[1] for c in self.chars.values()]

    @property
    def char_assignments(self) -> List[str]:
        return [c[0] for c in self.chars.values()]

    @property
    def char_images(self) -> np.ndarray:
        return np.stack([np.frombuffer(k, dtype=np.uint8)
                         for k in self.char_bytes]).reshape((-1, 16, 16))

    @property
    def confirmed_texts(self) -> Dict[str, Tuple[str, str]]:
        if not hasattr(self, "_confirmed_texts"):
            self._confirmed_texts: Dict[str, Tuple[str, str]] = {}
        return self._confirmed_texts

    @confirmed_texts.setter
    def confirmed_texts(self, value: Dict[str, Tuple[str, str]]) -> None:
        if not (isinstance(value, dict)):
            raise ValueError()
        self._confirmed_texts = value

    @property
    def confirmed_texts_languages(self) -> List[str]:
        return [t[0] for t in self.confirmed_texts.values()]

    @property
    def confirmed_texts_texts(self) -> List[str]:
        return [t[1] for t in self.confirmed_texts.values()]

    @property
    def dump_directory(self) -> Optional[str]:
        if not hasattr(self, "_dump_directory"):
            self._dump_directory: Optional[str] = None
        return self._dump_directory

    @dump_directory.setter
    def dump_directory(self, value: Optional[str]) -> None:
        if value is not None:
            value = expandvars(value)
            if not (isdir(value) and access(value, R_OK)):
                raise ValueError()
        self._dump_directory = value

    @property
    def event_handler(self) -> FileCreatedEventHandler:
        if not hasattr(self, "_event_handler"):
            self._event_handler = self.FileCreatedEventHandler(self)
        return self._event_handler

    @property
    def font(self) -> str:
        if not hasattr(self, "_font"):
            raise ValueError()
        return self._font

    @font.setter
    def font(self, value: str) -> None:
        self._font = value

    @property
    def fontsize(self) -> int:
        if not hasattr(self, "_fontsize"):
            raise ValueError()
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value: int) -> None:
        self._fontsize = value

    @property
    def language(self) -> str:
        if not hasattr(self, "_language"):
            self._language = "english"
        return self._language

    @language.setter
    def language(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError()
        value = value.lower()
        self._language = value

    @property
    def load_directory(self) -> Optional[str]:
        if not hasattr(self, "_load_directory"):
            self._load_directory: Optional[str] = None
        return self._load_directory

    @load_directory.setter
    def load_directory(self, value: Optional[str]) -> None:
        if value is not None:
            value = expandvars(value)
            # TODO: Create if possible
            if not (isdir(value) and access(value, W_OK)):
                raise ValueError()
        self._load_directory = value

    @property
    def model_file(self) -> Union[str, None]:
        if not hasattr(self, "_model_file"):
            self._model_file: Union[str, None] = None
        return self._model_file

    @model_file.setter
    def model_file(self, value: str) -> None:
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
        self._model_file = value

    @property
    def observer(self) -> Observer:
        if not hasattr(self, "_observer"):
            self._observer = Observer()
            self._observer.schedule(self.event_handler, self.dump_directory)
        return self._observer

    @property
    def overwrite(self) -> bool:
        if not hasattr(self, "_overwrite"):
            self._overwrite = False
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError()
        self._overwrite = value

    @property
    def scale(self) -> int:
        if not hasattr(self, "_scale"):
            self._scale = 4
        return self._scale

    @scale.setter
    def scale(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError()
        if value <= 1:
            raise ValueError()
        self._scale = value

    @property
    def scaled_chars(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_scaled_chars"):
            self.draw_scaled_chars()
        return self._scaled_chars

    @scaled_chars.setter
    def scaled_chars(self, value: Dict[str, np.ndarray]) -> None:
        if not isinstance(value, dict):
            raise ValueError()
        self.scaled_chars = value

    @property
    def unconfirmed_texts(self) -> Dict[str, Tuple[str, np.ndarray]]:
        if not hasattr(self, "_unconfirmed_texts"):
            self._unconfirmed_texts: Dict[str, Tuple[str, np.ndarray]] = {}
        return self._unconfirmed_texts

    @unconfirmed_texts.setter
    def unconfirmed_texts(self,
                          value: Dict[str, Tuple[str, np.ndarray]]) -> None:
        if not (isinstance(value, dict)):
            raise ValueError()
        self._unconfirmed_texts = value

    @property
    def unconfirmed_texts_languages(self) -> List[str]:
        return [t[0] for t in self.unconfirmed_texts.values()]

    @property
    def unconfirmed_texts_indexes(self) -> List[np.ndarray]:
        return [t[1] for t in self.unconfirmed_texts.values()]

    @property
    def verbosity(self) -> int:
        """int: Level of output to provide"""
        if not hasattr(self, "_verbosity"):
            self._verbosity = 1
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: int) -> None:
        if not isinstance(value, int) and value >= 0:
            raise ValueError()
        self._verbosity = value

    @property
    def watch(self) -> bool:
        if not hasattr(self, "_watch"):
            self._watch = False
        return self._watch

    @watch.setter
    def watch(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError()
        self._watch = value

    # endregion

    # region Public Methods

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
        self.unconfirmed_texts[filename] = (self.language,
                                            np.array(indexes, np.uint32))
        if self.is_text_confirmable(filename):
            self.confirm_text(filename)

    def confirm_text(self, filename: str) -> None:
        if filename in self.confirmed_texts:
            raise ValueError()
        elif filename not in self.unconfirmed_texts:
            raise ValueError()

        if self.verbosity >= 2:
            print(f"{filename}: confirmed")
        self.confirmed_texts[filename] = (self.language,
                                          self.get_text(filename))
        del self.unconfirmed_texts[filename]

    def draw_scaled_chars(self) -> None:
        scaled_chars = {}

        western_chars = np.array(list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        numeric_chars = np.array(list("0123456789"))

        total_diff = 0
        for char, (assignment, confirmed) in self.chars.items():
            if not confirmed:
                continue
            size = 16 * self.scale
            # debug = False
            # if assignment not in western_chars:
            #     if assignment not in numeric_chars:
            #         debug = True

            # Load original character image for alignment
            # orig_data = np.frombuffer(char, dtype=np.uint8).reshape(16, 16)
            # orig_image = Image.fromarray(orig_data).resize((size, size),
            #                                                Image.BICUBIC)
            # orig_data = np.array(orig_image)
            # if debug:
            #     print(orig_data)
            #     self.show_image(orig_image)

            # Draw scaled character image
            scaled_image = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(scaled_image)
            font = ImageFont.truetype(self.font, self.fontsize)
            width, height = draw.textsize(assignment, font=font)
            xy = ((size - width) / 2, (size - height) / 2)
            draw.text(xy, assignment, font=font, fill=255)
            scaled_data = np.array(scaled_image)
            # if debug:
            #     self.show_image(scaled_image)

            # Align
            #
            # offsets = range(-1 * max_offset, max_offset + 1)
            # best_diff = size * size * 255
            # best_offset = None
            # for offset in product(offsets, offsets):
            #     diff = orig_data.astype(np.int16) \
            #            - np.roll(scaled_data, offset, (0, 1)).astype(np.int16)
            #     diff = np.abs(diff).sum()
            #     if diff < best_diff:
            #         best_diff = diff
            #         best_offset = offset
            # scaled_data = np.roll(scaled_data, best_offset, (0, 1))
            # total_diff += best_diff
            #
            # if debug:
            #     print(f"Best offset for {assignment} is {best_offset}, "
            #           f"yielding {best_diff}")
            #     diff = orig_data.astype(np.int16) - scaled_data.astype(
            #         np.int16)
            #     diff = np.abs(diff).astype(np.uint8)
            #     self.show_image(Image.fromarray(diff))
            #     self.show_image(Image.fromarray(scaled_data))
            #     input("Enter to continue")

            scaled_chars[assignment] = scaled_data

        # print(f"Total diff between new images and scaled old: {total_diff}")
        self._scaled_chars = scaled_chars

    def get_text(self, filename: str) -> str:
        if filename not in self.unconfirmed_texts:
            raise ValueError()

        indexes = self.unconfirmed_texts[filename][1]
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

        indexes = self.unconfirmed_texts[filename][1]
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
                    # if a in []:
                    #     a = ""
                    #     c = False
                    self.chars[i.tobytes()] = (a, c)

            # Load unconfirmed texts
            if "texts/unconfirmed" in cache:
                filenames = [f.decode("UTF8") for f in
                             np.array(cache["texts/unconfirmed/filenames"])]
                languages = [l.decode("UTF8") for l in
                             np.array(cache["texts/unconfirmed/languages"])]
                indexes = np.array(cache["texts/unconfirmed/indexes"])
                for f, l, i in zip(filenames, languages, indexes):
                    self.unconfirmed_texts[f] = (l, i)

            # Load confirmed texts
            if "texts/confirmed" in cache:
                filenames = [f.decode("UTF8") for f in
                             np.array(cache["texts/confirmed/filenames"])]
                languages = [l.decode("UTF8") for l in
                             np.array(cache["texts/confirmed/languages"])]
                texts = [t.decode("UTF8") for t in
                         np.array(cache["texts/confirmed/texts"])]
                for f, l, t in zip(filenames, languages, texts):
                    self.confirmed_texts[f] = (l, t)

    def manually_assign_chars(self) -> None:
        size = 16 * self.scale

        # Load model
        if self.model_file is not None:
            model = keras.models.load_model(self.model_file)
        else:
            model = None

        # Loop over characters and assign
        i = 0
        n_unassigned = len([v for v in self.chars.values() if not v[1]])
        for char, (assignment, confirmed) in self.chars.items():
            if confirmed:
                continue
            i += 1
            data = np.frombuffer(char, dtype=np.uint8).reshape(16, 16)
            image = Image.fromarray(data).resize((size, size), Image.NEAREST)
            print(data)
            self.show_image(image)
            try:
                if model is not None:
                    # embed()
                    data = np.expand_dims(np.expand_dims(
                        data.astype(np.float16) / 255.0, axis=0), axis=3)
                    label_pred = model.predict(data)
                    char_pred = hanzi_chars[
                        np.array(np.argsort(label_pred, axis=1)[:, -1])]
                    print(char_pred[0])
                    assignment = self.input_prefill(
                        f"Assign character image {i + 1}/{n_unassigned} as:",
                        char_pred[0])
                else:
                    assignment = input(f"Assign character image "
                                       f"{i + 1}/{n_unassigned} as:")
                if assignment != "":
                    if self.verbosity >= 2:
                        print(f"Assigned character image {i + 1} as "
                              f"'{assignment}'")
                    self.chars[char] = (assignment, True)
            except UnicodeDecodeError as e:
                print(e)
                break
            except KeyboardInterrupt:
                break

        # Reassess unconfirmed texts
        for filename in list(self.unconfirmed_texts.keys()):
            if self.is_text_confirmable(filename):
                self.confirm_text(filename)

    def process_file(self, filename: str, save: bool = False) -> None:

        def backup(filename: str) -> None:
            if (self.backup_directory is not None
                    and not isfile(f"{self.backup_directory}/{filename}")):
                copyfile(f"{self.dump_directory}/{filename}",
                         f"{self.backup_directory}/{filename}")
                if self.verbosity >= 1:
                    print(f"{self.dump_directory}/{filename} backed up "
                          f"to {self.backup_directory}/{filename}")

        # If file is already confirmed, skip
        if filename in self.confirmed_texts:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: "
                      f"previously confirmed")
            backup(filename)
            return

        # If file is known and unconfirmed, try confirming
        if filename in self.unconfirmed_texts:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: "
                      f"previously unconfirmed")
            backup(filename)
            if self.is_text_confirmable(filename):
                self.confirm_text(filename)
            return

        # If file is not a text image, skip
        if not self.is_file_text_image(filename):
            if self.verbosity >= 3:
                print(f"{self.dump_directory}/{filename}: "
                      f"not a text image")
            return

        # If file is a new text image, add
        if self.verbosity >= 2:
            print(f"{self.dump_directory}/{filename}: "
                  f"new text image")
        backup(filename)
        self.add_text(filename)

        # Optionally save file immediately
        if save and filename in self.confirmed_texts:
            self.save_text(filename)

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
                cache.create_dataset("texts/unconfirmed/languages",
                                     data=[l.encode("UTF8") for l in
                                           self.unconfirmed_texts_languages],
                                     dtype="S24",
                                     chunks=True,
                                     compression="gzip")
                cache.create_dataset("texts/unconfirmed/indexes",
                                     data=np.stack(
                                         self.unconfirmed_texts_indexes),
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
                cache.create_dataset("texts/confirmed/languages",
                                     data=[l.encode("UTF8") for l in
                                           self.confirmed_texts_languages],
                                     dtype="S24",
                                     chunks=True,
                                     compression="gzip")
                cache.create_dataset("texts/confirmed/texts",
                                     data=[t.encode("UTF8") for t in
                                           self.confirmed_texts_texts],
                                     dtype="S1024",
                                     chunks=True,
                                     compression="gzip")

    def save_text(self, filename: str) -> None:
        if isfile(f"{self.load_directory}/{filename}") and not self.overwrite:
            return

        s = 16 * self.scale
        text_data = np.zeros((256 * self.scale, 256 * self.scale, 4),
                             np.uint8)
        x = y = 0

        for i, char in enumerate(self.confirmed_texts[filename][1]):
            char_data = self.scaled_chars[char]
            x = i % 16
            y = i // 16
            text_data[y * s:(y + 1) * s, x * s:(x + 1) * s, 3] = char_data
        image = Image.fromarray(text_data)
        image.save(f"{self.load_directory}/{filename}")
        if self.verbosity >= 1:
            print(f"{self.load_directory}/{filename} saved")

    # endregion

    # region Static Methods

    @staticmethod
    def input_prefill(prompt: str, prefill: str) -> str:
        """
        Prompts user for input with pre-filled text

        Args:
            prompt (str): Prompt to present to user
            prefill (str): Text to prefill for user

        Returns:
            str: Text inputted by user
        """

        def pre_input_hook() -> None:
            insert_text(prefill)
            redisplay()

        set_pre_input_hook(pre_input_hook)
        result = input(prompt)
        set_pre_input_hook()

        return result

    @staticmethod
    def show_image(image: Image.Image) -> None:
        try:
            from imgcat import imgcat

            imgcat(image)
        except ImportError:
            image.show()

    # endregion


#################################### MAIN #####################################
if __name__ == "__main__":
    OOT3DHDTextGenerator()()
