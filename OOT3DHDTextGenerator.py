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
from __future__ import annotations

from collections import OrderedDict
from os import R_OK, W_OK, X_OK, access, listdir, remove
from os.path import basename, dirname, expandvars, isdir, isfile
from pathlib import Path
from readline import insert_text, redisplay, set_pre_input_hook
from shutil import copyfile
from subprocess import DEVNULL, PIPE, Popen
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import yaml
from IPython import embed
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from tensorflow import keras
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


################################### CLASSES ###################################
class OOT3DHDTextGenerator:
    """
    Generates hi-res text images for The Legend of Zelda: Ocarina of Time 3D
    """

    # TODO: Command-Line Argument for conf file
    # TODO: Handle times via brute force
    # TODO: Error message text
    # TODO: License

    # region Classes

    class FileCreatedEventHandler(FileSystemEventHandler):  # type: ignore
        """
        Handles file creation in dump folder
        """

        def __init__(self, host) -> None:  # type: ignore
            """
            Initializes

            Args:
                host (OOT3DHDTextGenerator): Host to which files will be passed
            """
            self.host = host

        def on_created(self, event):  # type: ignore
            """
            Handles a file creation event

            Args:
                event: File creation event whose file to process
            """
            filename = basename(event.key[1])
            self.host.process_file(filename)

    # endregion

    # region Class Variables

    package_root: str = str(Path(__file__).parent.absolute())

    # endregion

    # region Builtins

    def __init__(self, conf_file: str = "conf_cn.yaml"):
        """
        Initializes

        Args:
            conf_file (str): file from which to load configuration
        """

        # Read configuration
        if not (isfile(conf_file) and access(conf_file, R_OK)):
            raise ValueError()
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)

        # General configuration
        self.verbosity = conf.get("verbosity", 1)

        # Input configuration
        self.dump_directory = conf.get(
            "dump", "$HOME/.local/share/citra-emu/dump/textures/"
                    "000400000008F900")
        self.cache_file = conf.get("cache", f"{self.package_root}/cmn-Hans.h5")
        self.model_file = conf.get("model", None)

        # Operation configuration
        self.size = conf.get("size", 64)
        self.font = ImageFont.truetype(
            conf.get("font", "/System/Library/Fonts/STHeiti Medium.ttc"),
            conf.get("fontsize", 62))
        self.interactive = conf.get("interactive", False)
        self.xbrzscale = conf.get("xbrzscale", None)
        self.watch = conf.get("watch", False)

        # Output configuration
        self.load_directory = conf.get(
            "load",
            "load: $HOME/.local/share/citra-emu/load/textures/"
            "000400000008F900)")
        self.overwrite = conf.get("overwrite", False)
        self.backup_directory = conf.get("backup", None)

    def __call__(self) -> None:
        """
        Performs operations
        """

        # Load cache
        if isfile(self.cache_file):
            if self.verbosity >= 1:
                print(f"Loading cache from '{self.cache_file}'")
            self.load_hdf5_cache()

        # Review all existing images
        if self.verbosity >= 1:
            print(f"Loading images from  '{self.dump_directory}'")
        for filename in listdir(self.dump_directory):
            self.process_file(filename)

        # Assign unassigned characters
        if self.interactive:
            self.interactively_assign_chars()
            self._initialize_hires_char_images()

        # Save cache
        if self.verbosity >= 1:
            print(f"Saving cache to '{self.cache_file}'")
        self.save_hdf5_cache()

        # Save text images
        if self.verbosity >= 1:
            print(f"Saving images to '{self.load_directory}'")
        for filename in self.assigned_files:
            self.save_hires_file(filename)

        # Watch for additional images and process as they appear
        if self.watch:
            if self.verbosity >= 1:
                print(f"Watching for new images in '{self.dump_directory}'")
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
        self.save_hdf5_cache()

    # endregion

    # region Properties

    @property
    def assigned_files(self) -> Dict[str, str]:
        """Dict[str, str]: Lo-res image files and assigned text content"""
        if not hasattr(self, "_assigned_files"):
            self._assigned_files: Dict[str, str] = {}
        return self._assigned_files

    @assigned_files.setter
    def assigned_files(self, value: Dict[str, str]) -> None:
        if not (isinstance(value, dict)):
            raise ValueError()
        self._assigned_files = value

    @property
    def backup_directory(self) -> Optional[str]:
        """Optional[str]: Directory to which to back up lo-res image files"""
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
    def cache_file(self) -> str:
        """Optional[str]: HDF5 cache file path"""
        if not hasattr(self, "_cache_file"):
            raise ValueError()
        return self._cache_file

    @cache_file.setter
    def cache_file(self, value: str) -> None:
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
    def lores_char_images(self) -> OrderedDict[bytes, str]:
        """OrderedDict[bytes, str]: Lo-res character images and assignments"""
        if not hasattr(self, "_lores_char_images"):
            self._lores_char_images: OrderedDict[bytes, str] = OrderedDict()
        return self._lores_char_images

    @lores_char_images.setter
    def lores_char_images(self, value: OrderedDict[bytes, str]) -> None:
        if not (isinstance(value, OrderedDict)):
            raise ValueError()
        self._lores_char_images = value

    @property
    def lores_char_assignments(self) -> List[str]:
        """List[str]: Characters assigned to character images"""
        return list(self.lores_char_images.values())

    @property
    def lores_char_bytes(self) -> List[bytes]:
        """List[bytes]: Character images in byte form"""
        return list(self.lores_char_images.keys())

    @property
    def lores_char_array(self) -> np.ndarray:
        """numpy.ndarray: Character images in numpy array form"""
        return np.stack(
            [np.frombuffer(k, dtype=np.uint8) for k in
             self.lores_char_images.keys()]
        ).reshape((-1, 16, 16))

    @property
    def dump_directory(self) -> Optional[str]:
        """Optional[str]: Directory from which to load lo-res image files"""
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
        """FileCreatedEventHandler: Handler for files in dump directory"""
        if not hasattr(self, "_event_handler"):
            self._event_handler = self.FileCreatedEventHandler(self)
        return self._event_handler

    @property
    def font(self) -> ImageFont.truetype:
        """ImageFont.truetype: Font for hi-res images"""
        if not hasattr(self, "_font"):
            self._font: ImageFont.truetype = ImageFont.truetype(
                "/System/Library/Fonts/STHeiti Medium.ttc", 62)
        return self._font

    @font.setter
    def font(self, value: ImageFont.truetype) -> None:
        self._font = value

    @property
    def hires_char_images(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: hi-res images of characters"""
        if not hasattr(self, "_hires_char_images"):
            self._initialize_hires_char_images()
        return self._hires_char_images

    @hires_char_images.setter
    def hires_char_images(self, value: Dict[str, np.ndarray]) -> None:
        if not isinstance(value, dict):
            raise ValueError()
        self.hires_char_images = value

    @property
    def interactive(self) -> bool:
        """bool: Interactively assign unassigned character images"""
        if not hasattr(self, "_interactive"):
            self._interactive = False
        return self._interactive

    @interactive.setter
    def interactive(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError()
        self._interactive = value

    @property
    def load_directory(self) -> Optional[str]:
        """Optional[str]: Directory to which to save hi-res image files"""
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
    def model_file(self) -> Optional[str]:
        """Optional[str]: Model to use for lo-res character assignment"""
        if not hasattr(self, "_model_file"):
            self._model_file: Optional[str] = None
        return self._model_file

    @model_file.setter
    def model_file(self, value: Optional[str]) -> None:
        if value is not None:
            value = expandvars(value)
            if isfile(value):
                if not (access(value, R_OK) and access(value, W_OK)):
                    raise ValueError()
            else:
                raise ValueError
        self._model_file = value

    @property
    def observer(self) -> Observer:
        """Observer: Observer of new files in dump directory"""
        if not hasattr(self, "_observer"):
            self._observer = Observer()
            self._observer.schedule(self.event_handler, self.dump_directory)
        return self._observer

    @property
    def overwrite(self) -> bool:
        """bool: Overwrite existing hi-res image files in load directory"""
        if not hasattr(self, "_overwrite"):
            self._overwrite = False
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError()
        self._overwrite = value

    @property
    def size(self) -> int:
        """int: Height and width of hi-res character images"""
        if not hasattr(self, "_size"):
            self._size = 4
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError()
        if not (np.log2(value).is_integer() and np.log2(value) > 5):
            raise ValueError()
        self._size = value

    @property
    def unassigned_files(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Lo-res image files and character indexes"""
        if not hasattr(self, "_unassigned_files"):
            self._unassigned_files: Dict[str, np.ndarray] = {}
        return self._unassigned_files

    @unassigned_files.setter
    def unassigned_files(self, value: Dict[str, np.ndarray]) -> None:
        if not (isinstance(value, dict)):
            raise ValueError()
        self._unassigned_files = value

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
        """bool: Watch for new image files after processing existing"""
        if not hasattr(self, "_watch"):
            self._watch = False
        return self._watch

    @watch.setter
    def watch(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError()
        self._watch = value

    @property
    def xbrzscale(self) -> Optional[str]:
        """Optional[str]: xbrzscale executable file path"""
        if not hasattr(self, "_xbrzscale"):
            raise ValueError()
        return self._xbrzscale

    @xbrzscale.setter
    def xbrzscale(self, value: str) -> None:
        value = expandvars(value)
        if not isfile(value) or not (access(value, X_OK)):
            raise ValueError()
        self._xbrzscale = value

    # endregion

    # region Public Methods

    def backup_file(self, filename: str) -> None:
        """
        Copies a lo-res image file from the dump to the backup directory

        Args:
            filename (str): image file to back up
        """
        if self.backup_directory is None:
            return
        elif isfile(f"{self.backup_directory}/{filename}"):
            return

        copyfile(f"{self.dump_directory}/{filename}",
                 f"{self.backup_directory}/{filename}")
        if self.verbosity >= 1:
            print(f"{self.dump_directory}/{filename}: backed up "
                  f"to {self.backup_directory}/{filename}")

    def assign_file(self, filename: str) -> None:
        """
        Marks an image file as having its text content assigned

        Args:
            filename (str): image file to assign
        """
        if filename not in self.unassigned_files:
            raise ValueError()
        elif filename in self.assigned_files:
            return

        indexes = self.unassigned_files[filename]
        assignments = [self.lores_char_assignments[i] for i in indexes]

        if "" in assignments:
            return

        self.assigned_files[filename] = "".join(assignments).rstrip()
        del self.unassigned_files[filename]
        if self.verbosity >= 1:
            print(f"{filename}: assigned")

    def load_hdf5_cache(self) -> None:
        """
        Loads character and image file data structures from hdf5 cache
        """
        with h5py.File(self.cache_file) as cache:

            # Load characters
            if "characters" in cache:
                images = np.array(cache["characters/images"])
                assignments = np.array(cache["characters/assignments"])
                for i, a in zip(images, assignments):
                    try:
                        a = a.decode("UTF8")
                    except UnicodeDecodeError as e:
                        print(f"Error encountered while decoding characters: "
                              f"{e}, for image:")
                        print(i)
                        print("skipping...")
                        sleep(1)
                    # if a in []:
                    #     a = ""
                    self.lores_char_images[i.tobytes()] = a

            # Load unassigned texts
            if "files/unassigned" in cache:
                filenames = [f.decode("UTF8") for f in
                             np.array(cache["files/unassigned/filenames"])]
                indexes = np.array(cache["files/unassigned/indexes"])
                for f, i in zip(filenames, indexes):
                    self.unassigned_files[f] = i

            # Load assigned texts
            if "files/assigned" in cache:
                filenames = [f.decode("UTF8") for f in
                             np.array(cache["files/assigned/filenames"])]
                texts = [t.decode("UTF8") for t in
                         np.array(cache["files/assigned/texts"])]
                for f, t in zip(filenames, texts):
                    self.assigned_files[f] = t

    def load_file(self, filename: str) -> None:
        """
        Loads a lo-res image file from the dump directory

        Adds any new character images to self.chars. Initially adds text to
        self.unassigned_files, and if all characters are assigned calls
        self.assign_file to move to self.assigned_files

        Args:
            filename (str): lo-res image file to add
        """
        if filename in self.unassigned_files:
            return
        elif filename in self.assigned_files:
            return

        text_data = np.array(Image.open(f"{self.dump_directory}/{filename}"))
        text_data = text_data[:, :, 3]

        indexes = []
        for x in range(16):
            for y in range(16):
                char_data = text_data[x * 16:(x + 1) * 16, y * 16:(y + 1) * 16]
                char_bytes = char_data.tobytes()
                if char_bytes not in self.lores_char_images:
                    self.lores_char_images[char_bytes] = ""
                indexes.append(self.lores_char_bytes.index(char_bytes))

        self.unassigned_files[filename] = np.array(indexes, np.uint32)
        if self.verbosity >= 1:
            print(f"{filename}: added")

        self.assign_file(filename)

    def interactively_assign_chars(self) -> None:
        """
        Prompts user to interactively assign unassigned character images
        """

        # Load model
        if self.model_file is not None:
            model = keras.models.load_model(self.model_file)
            hanzi_frequency: pd.DataFrame = pd.read_csv(
                f"{self.package_root}/data/characters.txt",
                sep="\t",
                names=["character", "frequency", "cumulative frequency"])
            hanzi_chars: List[str] = hanzi_frequency["character"].tolist()

        else:
            model = None

        def get_predicted_assignment(data_uint8: np.ndarray) -> str:
            data_float16 = np.expand_dims(np.expand_dims(
                data_uint8.astype(np.float16) / 255.0, axis=0), axis=3)
            predicted_index = model.predict(data_float16)
            return str(
                hanzi_chars[np.argsort(predicted_index, axis=1)[:, -1][0]])

        def get_xbrz_image(data_uint8: np.ndarray) -> Image.Image:
            lores_tempfile = NamedTemporaryFile(delete=False, suffix=".png")
            Image.fromarray(data_uint8).save(lores_tempfile)
            lores_tempfile.close()

            xbrz_tempfile = NamedTemporaryFile(delete=False, suffix=".png")
            xbrz_tempfile.close()
            command = f"{self.xbrzscale} 6 " \
                      f"{lores_tempfile.name} " \
                      f"{xbrz_tempfile.name}"
            Popen(command, shell=True, stdin=PIPE, stdout=DEVNULL,
                  stderr=DEVNULL, close_fds=True).wait()
            image = Image.open(xbrz_tempfile.name)
            remove(lores_tempfile.name)
            remove(xbrz_tempfile.name)
            return image

        def get_nn_image(data_uint8: np.ndarray) -> Image.Image:
            return Image.fromarray(data_uint8).resize((self.size, self.size),
                                                      Image.NEAREST)

        # Loop over characters and assign
        i = 0
        unassigned_chars = OrderedDict(
            [(i, a) for i, a in self.lores_char_images.items() if a == ""])
        for char_bytes, assignment in unassigned_chars.items():
            i += 1
            char_data = np.frombuffer(char_bytes, dtype=np.uint8).reshape(
                16, 16)

            if self.xbrzscale is not None:
                char_image = get_xbrz_image(char_data)
            else:
                char_image = get_nn_image(char_data)
            print()
            self.show_image(char_image)
            print()
            try:
                if model is not None:
                    assignment = self.input_prefill(
                        f"Assign character image "
                        f"{i + 1}/{len(unassigned_chars)} as:",
                        get_predicted_assignment(char_data))
                else:
                    assignment = input(
                        f"Assign character image "
                        f"{i + 1}/{len(unassigned_chars)} as:")
                if assignment != "":
                    if self.verbosity >= 1:
                        print(f"Assigned character image as '{assignment}'")
                    self.lores_char_images[char_bytes] = assignment
            except UnicodeDecodeError as e:
                print(e)
                break
            except KeyboardInterrupt:
                break

        # Reassess unassigned texts
        for filename in list(self.unassigned_files.keys()):
            self.assign_file(filename)

    def process_file(self, filename: str) -> None:
        """
        Processes a lo-res image file in the dump directory

        Checks if file is a known text image, not a text image, or a new text
        image.

        Args:
            filename (str): lo-res image file to process
        """

        def is_text_image_file(filename: str) -> bool:
            """
            Checks if a file within the dump directory is a text image file

            Args:
                filename (str): file to check

            Returns:
                bool: True if file is a text image file; false otherwise
            """
            try:
                image = Image.open(f"{self.dump_directory}/{filename}")
                data = np.array(image)
            except UnidentifiedImageError:
                return False

            if data.shape != (256, 256, 4):
                return False
            if data[:, :, :3].sum() != 0:
                return False

            return True

        # If file is already assigned, skip
        if filename in self.assigned_files:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: previously "
                      f"assigned")

        # If file is known and unassigned, try assigning
        elif filename in self.unassigned_files:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: previously "
                      f"unassigned")
            self.assign_file(filename)

        # If file is not a text image, skip
        elif not is_text_image_file(filename):
            if self.verbosity >= 3:
                print(f"{self.dump_directory}/{filename}: not a text image "
                      f"file")
            return

        # If file is a new text image, load
        else:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: new text image file")
            self.load_file(filename)

        # Back up lo-res and save hi-res file
        self.backup_file(filename)
        if filename in self.assigned_files:
            self.save_hires_file(filename)

    def save_hdf5_cache(self) -> None:
        """
        Saves character and text image file data structures to hdf5 cache
        """
        with h5py.File(self.cache_file) as cache:

            # Save characters
            if "characters" in cache:
                del cache["characters"]
            cache.create_dataset(
                "characters/images",
                data=self.lores_char_array,
                dtype=np.uint8,
                chunks=True,
                compression="gzip")
            cache.create_dataset(
                "characters/assignments",
                data=[a.encode("UTF8") for a in self.lores_char_assignments],
                dtype="S4",
                chunks=True,
                compression="gzip")

            # Save unassigned texts
            if "files/unassigned" in cache:
                del cache["files/unassigned"]
            if len(self.unassigned_files) > 0:
                cache.create_dataset(
                    "files/unassigned/filenames",
                    data=[k.encode("UTF8") for k in
                          self.unassigned_files.keys()],
                    dtype="S48",
                    chunks=True,
                    compression="gzip")
                cache.create_dataset(
                    "files/unassigned/indexes",
                    data=np.stack(list(
                        self.unassigned_files.values())),
                    dtype=np.uint32,
                    chunks=True,
                    compression="gzip")

            # Save assigned texts
            if "files/assigned" in cache:
                del cache["files/assigned"]
            if len(self.assigned_files) > 0:
                cache.create_dataset(
                    "files/assigned/filenames",
                    data=[k.encode("UTF8") for k in
                          self.assigned_files.keys()],
                    dtype="S48",
                    chunks=True,
                    compression="gzip")
                cache.create_dataset(
                    "files/assigned/texts",
                    data=[t.encode("UTF8") for t in
                          self.assigned_files.values()],
                    dtype="S1024",
                    chunks=True,
                    compression="gzip")

    def save_hires_file(self, filename: str) -> None:
        """
        Saves a hi-res image file to the load directory

        Args:
            filename (str): text image file to save
        """
        if isfile(f"{self.load_directory}/{filename}") and not self.overwrite:
            return

        text_data = np.zeros((16 * self.size, 16 * self.size, 4), np.uint8)
        for i, char in enumerate(self.assigned_files[filename]):
            char_data = self.hires_char_images[char]
            x = i % 16
            y = i // 16
            # @formatter:off
            text_data[y * self.size:(y + 1) * self.size,
                      x * self.size:(x + 1) * self.size, 3] = char_data
            # @formatter:on

        Image.fromarray(text_data).save(f"{self.load_directory}/{filename}")
        if self.verbosity >= 1:
            print(f"{self.load_directory}/{filename} saved")

    # endregion

    # region Private Methods

    def _initialize_hires_char_images(self) -> None:
        """
        Initializes hi-res character images
        """
        hires_char_images = {}

        for assignment in [a for a in self.lores_char_assignments if a != ""]:
            hires_image = Image.new("L", (self.size, self.size), 0)
            draw = ImageDraw.Draw(hires_image)
            width, height = draw.textsize(assignment, font=self.font)
            draw.text(((self.size - width) / 2, (self.size - height) / 2),
                      assignment, font=self.font, fill=255)

            hires_char_images[assignment] = np.array(hires_image)

        self._hires_char_images = hires_char_images

    # endregion

    # region Static Methods

    @staticmethod
    def input_prefill(prompt: str, prefill: str) -> str:
        """
        Prompts user for input with pre-filled text

        Args:
            prompt (str): prompt to present to user
            prefill (str): text to prefill for user

        Returns:
            str: text input by user
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
        """
        Shows an image to the user, in the terminal if possible

        Args:
            image (Image.Image): image to show
        """
        try:
            from imgcat import imgcat

            imgcat(image)
        except ImportError:
            image.show()

    # endregion


#################################### MAIN #####################################
if __name__ == "__main__":
    OOT3DHDTextGenerator()()
