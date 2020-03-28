#!python
#   OOT3DHDTextGenerator.py
#
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license.
################################### MODULES ###################################
from __future__ import annotations

import re
from argparse import ArgumentError, ArgumentParser, RawDescriptionHelpFormatter
from collections import OrderedDict
from itertools import product
from os import R_OK, W_OK, X_OK, access, getcwd, listdir, remove
from os.path import basename, dirname, expandvars, isdir, isfile
from pathlib import Path
from readline import insert_text, redisplay, set_pre_input_hook
from shutil import copyfile
from subprocess import DEVNULL, PIPE, Popen
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFont, UnidentifiedImageError


################################### CLASSES ###################################
class OOT3DHDTextGenerator:
    """
    Generates hi-res text images for The Legend of Zelda: Ocarina of Time 3D

    TODO:
        - [x] Command-Line Argument for conf file
        - [x] Handle 512 x 128 and 256 x 128 text
        - [ ] Add License
        - [x] Assign remaining characters
        - [ ] Sort characters
        - [ ] Review assignments
        - [ ] Review hires image output logic
        - [ ] Reconsider how to handle model
        - [ ] Document
        - [ ] Add useful error message text
        - [ ] Add requirements file
        - [ ] Include a free font
        - [ ] Test on Windows
        - [ ] Handle times via brute force
        - [ ] Bilingual README
    """

    package_root: str = str(Path(__file__).parent.absolute())
    re_filename = re.compile(
        "tex1_(?P<width>\d+)x(?P<height>\d+)_(?P<id>[^_]+)_(?P<kind>\d+)\.png")

    # region Builtins

    def __init__(self, conf_file: str, verbosity: int = 1):
        """
        Initializes

        Args:
            conf_file (str): file from which to load configuration
        """

        # General configuration
        self.verbosity = verbosity

        # Read configuration file
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)

        # Input configuration
        self.cache_file = conf.get("cache", f"{self.package_root}/cmn-Hans.h5")
        self.dump_directory = conf.get(
            "dump",
            "$HOME/.local/share/citra-emu/dump/textures/000400000008F900")
        self.model = conf.get("model", None)

        # Operation configuration
        self.size = conf.get("size", 64)
        self.font = ImageFont.truetype(
            conf.get("font", "/System/Library/Fonts/STHeiti Medium.ttc"),
            conf.get("fontsize", 62))
        self.xbrzscale = conf.get("xbrzscale", None)
        self.operations["scan"] = conf.get("scan", True)
        self.operations["offset"] = conf.get("offset", (0, 0))
        self.operations["align"] = conf.get("align", False)
        self.operations["assign"] = conf.get("assign", False)
        self.operations["validate"] = conf.get("validate", False)
        self.operations["watch"] = conf.get("watch", False)

        # Output configuration
        self.load_directory = conf.get(
            "load",
            "$HOME/.local/share/citra-emu/load/textures/000400000008F900)")
        self.operations["overwrite"] = conf.get("overwrite", False)
        self.backup_directory = conf.get("backup", None)

    def __call__(self) -> None:
        """
        Performs operations
        """

        # Load cache
        if isfile(self.cache_file):
            self.load_hdf5_cache()

        # Write known images
        for filename in list(self.assigned_files):
            self.save_hires_file(filename)

        # Scan existing images
        if self.operations["scan"]:
            n_files = self.scan_dump_directory()
            if n_files > 0:
                self.save_hdf5_cache()

        # Assign unassigned characters
        if self.operations["assign"]:
            n_assigned = self.interactively_assign_chars()
            if n_assigned > 0:
                if self.verbosity >= 1:
                    print(f"Assigned {n_assigned} characters, "
                          f"reassessing unassigned files")
                for filename in list(self.unassigned_files.keys()):
                    self.assign_file(filename)
                self.save_hdf5_cache()
                self.hires_chars = {}
                self.save_load_directory()

        # Validate assigned characters
        if self.operations["validate"]:
            n_reassigned = self.interactively_validate_chars()
            if n_reassigned > 0:
                if self.verbosity >= 1:
                    print(f"Reassigned {n_reassigned} characters, "
                          f"reassessing all files")
                self.assigned_files = {}
                self.unassigned_files = {}
                if self.operations["scan"]:
                    self.scan_dump_directory()
                self.save_hdf5_cache()
                self.hires_chars = {}
                self.save_load_directory()

        # Watch for additional images
        if self.operations["watch"]:
            n_files = self.watch_dump_directory()
            if n_files > 0:
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
        if dirname(value) == "":
            value = f"{getcwd()}/{value}"

        if isfile(value):
            if not (access(value, R_OK) and access(value, W_OK)):
                raise ValueError()
        elif isdir(dirname(value)):
            if not (access(dirname(value), R_OK)
                    and access(dirname(value), W_OK)):
                raise ValueError()
        else:
            raise ValueError()

        self._cache_file = value

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
    def hanzi_chars(self) -> List[str]:
        if not hasattr(self, "_hanzi_chars"):
            self._hanzi_chars: List[str] = list(np.loadtxt(
                f"{self.package_root}/data/characters.txt",
                dtype=str, usecols=0))
        return self._hanzi_chars

    @property
    def hires_chars(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: hi-res images of characters"""
        if not hasattr(self, "_hires_chars") or self._hires_chars == {}:
            hires_chars = {}

            lefts = []
            rights = []
            tops = []
            bottoms = []
            for i, assignment in enumerate(self.lores_char_assignments):
                if assignment == "":
                    continue
                hires_image = Image.new("L", (self.size, self.size), 0)
                draw = ImageDraw.Draw(hires_image)
                width, height = draw.textsize(assignment, font=self.font)
                draw.text((((self.size - width) / 2)
                           + self.operations["offset"][0],
                           ((self.size - height) / 2)
                           + self.operations["offset"][1]),
                          assignment, font=self.font, fill=255, align="center")
                hires_data = np.array(hires_image)

                try:
                    transparent_cols = list((hires_data == 0).all(axis=0))
                    lefts.append(transparent_cols.index(False))
                    rights.append(
                        list(reversed(transparent_cols)).index(False))
                    transparent_rows = list((hires_data == 0).all(axis=1))
                    tops.append(transparent_rows.index(False))
                    bottoms.append(
                        list(reversed(transparent_rows)).index(False))
                except ValueError:
                    pass

                if self.operations["align"]:
                    lores_data = self.lores_char_array[i]
                    scaled_data = np.array(self.get_scaled_image(lores_data))

                    best_diff = self.size * self.size * 255
                    best_offset = None
                    for offset in product(range(-8, 9), range(-8, 9)):
                        diff = scaled_data.astype(np.int16) \
                               - np.roll(hires_data, offset,
                                         (0, 1)).astype(np.int16)
                        diff = np.abs(diff).sum()
                        if diff < best_diff:
                            best_diff = diff
                            best_offset = offset
                    hires_data = np.roll(hires_data, best_offset, (0, 1))

                hires_chars[assignment] = hires_data

            if self.verbosity >= 1:
                print(f"Minimum buffer around edges: left = {min(lefts)}, "
                      f"right = {min(rights)}, top = {min(tops)}, "
                      f"bottom = {min(bottoms)}")

            self._hires_chars: Dict[str, np.ndarray] = hires_chars

        return self._hires_chars

    @hires_chars.setter
    def hires_chars(self, value: Dict[str, np.ndarray]) -> None:
        if not isinstance(value, dict):
            raise ValueError()
        self._hires_chars = value

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
    def lores_chars(self) -> OrderedDict[bytes, str]:
        """OrderedDict[bytes, str]: Lo-res character bytes and assignments"""
        if not hasattr(self, "_lores_chars"):
            self._lores_chars: OrderedDict[bytes, str] = OrderedDict()
        return self._lores_chars

    @lores_chars.setter
    def lores_chars(self, value: OrderedDict[bytes, str]) -> None:
        if not (isinstance(value, OrderedDict)):
            raise ValueError()
        self._lores_chars = value

    @property
    def lores_char_assignments(self) -> List[str]:
        """List[str]: Characters assigned to character images"""
        return list(self.lores_chars.values())

    @property
    def lores_char_bytes(self) -> List[bytes]:
        """List[bytes]: Character images in byte form"""
        return list(self.lores_chars.keys())

    @property
    def lores_char_array(self) -> np.ndarray:
        """numpy.ndarray: Character images in numpy array form"""
        return np.stack(
            [np.frombuffer(k, dtype=np.uint8) for k in
             self.lores_chars.keys()]
        ).reshape((-1, 16, 16))

    @property
    def model(self):  # type: ignore
        """Optional[keras.Sequential]: Lo-res character assignment model"""
        if not hasattr(self, "_model"):
            self._model = None
        return self._model

    @model.setter
    def model(self, value) -> None:  # type: ignore
        if value is not None:
            try:
                from tensorflow.keras.models import load_model
            except ImportError as e:
                raise e

            value = expandvars(value)
            if isfile(value):
                if not (access(value, R_OK) and access(value, W_OK)):
                    raise ValueError()
            else:
                raise ValueError
            value = load_model(value)
        self._model = value

    @property
    def operations(self) -> Dict[str, Any]:
        """Dict[str, Any]: Operations to perform and associated flags"""
        if not hasattr(self, "_operations"):
            self._operations: Dict[str, Any] = {}
        return self._operations

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

    # region Methods

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

    def get_predicted_assignment(self, data_uint8: np.ndarray) -> str:
        if self.model is None:
            raise ValueError()
        data_float16 = np.expand_dims(np.expand_dims(
            data_uint8.astype(np.float16) / 255.0, axis=0), axis=3)
        predicted_index = self.model.predict(data_float16)
        return str(
            self.hanzi_chars[np.argsort(predicted_index, axis=1)[:, -1][0]])

    def get_scaled_image(self, data_uint8: np.ndarray) -> Image.Image:
        if self.xbrzscale is None:
            return Image.fromarray(data_uint8).resize(
                (self.size, self.size), Image.NEAREST)

        lores_tempfile = NamedTemporaryFile(delete=False, suffix=".png")
        Image.fromarray(data_uint8).save(lores_tempfile)
        lores_tempfile.close()

        xbrz_tempfile = NamedTemporaryFile(delete=False, suffix=".png")
        xbrz_tempfile.close()

        command = f"{self.xbrzscale} 4 " \
                  f"{lores_tempfile.name} " \
                  f"{xbrz_tempfile.name}"
        Popen(command, shell=True, stdin=PIPE, stdout=DEVNULL,
              stderr=DEVNULL, close_fds=True).wait()

        image = Image.open(xbrz_tempfile.name).convert("L")

        remove(lores_tempfile.name)
        remove(xbrz_tempfile.name)

        return image

    def interactively_assign_chars(self) -> int:
        """
        Prompts user to interactively assign unassigned character images

        Returns:
            int: number of characters assigned
        """
        unassigned_chars = [b for b, a in self.lores_chars.items() if a == ""]
        n_unassigned = len(unassigned_chars)
        n_assigned = 0

        if self.verbosity >= 1:
            print(f"Interactively assigning {n_unassigned} characters")
            print("CTRL-C to exit; CTRL-D to undo")
        i = 0
        while i < n_unassigned:
            # Gather data and generate images
            lores_char_bytes = unassigned_chars[i]
            lores_char_data = np.frombuffer(
                lores_char_bytes, dtype=np.uint8).reshape(16, 16)
            lores_char_image = self.get_scaled_image(lores_char_data)

            # Prompt for assignment
            self.show_image(lores_char_image)
            try:
                if self.model is not None:
                    assignment = self.input_prefill(
                        f"Assign character {i}/{n_unassigned} as:",
                        self.get_predicted_assignment(lores_char_data))
                else:
                    assignment = input(
                        f"Assign character {i}/{n_unassigned} as:")
            except EOFError:
                print()
                if i != 0:
                    i -= 1
                continue
            except KeyboardInterrupt:
                print()
                print("Quitting interactive assignment")
                break

            # Assign
            if len(assignment) == 1:
                self.lores_chars[lores_char_bytes] = assignment
                if self.verbosity >= 1:
                    print(f"Assigned character as '{assignment}'")
                n_assigned += 1
            i += 1

        return n_assigned

    def interactively_validate_chars(self) -> int:
        """
        Prompts user to interactively validate assigned character images

        Returns:
            int: number of characters reassigned
        """
        n_assigned = len(self.hires_chars)
        n_reassigned = 0

        if self.verbosity >= 1:
            print(f"Interactively validating {n_assigned} character "
                  f"assignments")
            print("CTRL-C to exit; CTRL-D to undo")
        i = 0
        while i < len(self.lores_chars):
            # Gather data and generate images
            assignment = self.lores_char_assignments[i]
            if assignment == "":
                i += 1
                continue
            lores_char_bytes = self.lores_char_bytes[
                self.lores_char_assignments.index(assignment)]
            lores_char_data = np.frombuffer(
                lores_char_bytes, dtype=np.uint8).reshape(16, 16)
            lores_char_image = self.get_scaled_image(lores_char_data)
            hires_char_data = self.hires_chars[assignment]
            hires_char_image = Image.fromarray(hires_char_data)
            diff_image = ImageChops.difference(lores_char_image,
                                               hires_char_image)
            concatenated_image = self.concatenate_images(
                lores_char_image, hires_char_image, diff_image)

            # Prompt for reassignment
            self.show_image(concatenated_image)
            try:
                new_assignment = self.input_prefill(
                    f"Character {i}/{n_assigned} assigned as:",
                    assignment)
            except EOFError:
                print()
                if i != 0:
                    i -= 1
                continue
            except KeyboardInterrupt:
                print()
                print("Quitting interactive validation")
                break

            # Reassign
            if new_assignment != assignment and len(assignment) == 1:
                self.lores_chars[lores_char_bytes] = new_assignment
                if self.verbosity >= 1:
                    print(f"Reassigned character as '{new_assignment}'")
                n_reassigned += 1
            i += 1

        return n_reassigned

    def load_hdf5_cache(self) -> None:
        """
        Loads character and image file data structures from hdf5 cache
        """
        if self.verbosity >= 1:
            print(f"Loading cache from '{self.cache_file}'")
        with h5py.File(self.cache_file) as cache:

            # Load characters
            if "characters" in cache:
                images = np.array(cache["characters/images"])
                assignments = np.array(cache["characters/assignments"])
                for i, a in zip(images, assignments):
                    self.lores_chars[i.tobytes()] = a.decode("UTF8")

            # Load unassigned texts
            if "files/unassigned" in cache:
                filenames = np.array(cache["files/unassigned/filenames"])
                indexes = np.array(cache["files/unassigned/indexes"])
                for f, i in zip(filenames, indexes):
                    self.unassigned_files[f.decode("UTF8")] = i

            # Load assigned texts
            if "files/assigned" in cache:
                filenames = np.array(cache["files/assigned/filenames"])
                texts = np.array(cache["files/assigned/texts"])
                for f, t in zip(filenames, texts):
                    self.assigned_files[f.decode("UTF8")] = t.decode("UTF8")

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
        for y in range(text_data.shape[0] // 16):
            for x in range(text_data.shape[1] // 16):
                char_data = text_data[y * 16:(y + 1) * 16, x * 16:(x + 1) * 16]
                char_bytes = char_data.tobytes()
                if char_bytes not in self.lores_chars:
                    self.lores_chars[char_bytes] = ""
                indexes.append(self.lores_char_bytes.index(char_bytes))

        # For images with less than 256 characters, extend with spaces
        if len(indexes) != 256:
            indexes.extend([self.lores_char_bytes.index(
                np.zeros((16, 16), np.uint8).tobytes())] * 128)

        self.unassigned_files[filename] = np.array(indexes, np.uint32)
        if self.verbosity >= 1:
            print(f"{filename}: added")

        self.assign_file(filename)

    def process_file(self, filename: str) -> bool:
        """
        Processes a lo-res image file in the dump directory

        Checks if file is a known text image, not a text image, or a new text
        image.

        Args:
            filename (str): lo-res image file to process

        Returns:
            bool: True if file is a new text image file
        """

        def is_text_image_file(filename: str) -> bool:
            """
            Checks if a file within the dump directory is a text image file

            Args:
                filename (str): file to check

            Returns:
                bool: True if file is a text image file; false otherwise
            """
            match = self.re_filename.match(filename)
            if not match:
                return False
            elif int(match["kind"]) != 11:
                return False
            try:
                image = Image.open(f"{self.dump_directory}/{filename}")
                data = np.array(image)
            except UnidentifiedImageError:
                return False
            if data.shape not in [(128, 256, 4), (128, 512, 4), (256, 256, 4)]:
                return False
            if data[:, :, :3].sum() != 0:
                return False

            return True

        # If file is already assigned, skip
        if filename in self.assigned_files:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: previously "
                      f"assigned")
            new_file = False

        # If file is known and unassigned, try assigning
        elif filename in self.unassigned_files:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: previously "
                      f"unassigned")
            new_file = False
            self.assign_file(filename)

        # If file is not a text image, skip
        elif not is_text_image_file(filename):
            if self.verbosity >= 3:
                print(f"{self.dump_directory}/{filename}: not a text image "
                      f"file")
            return False

        # If file is a new text image, load
        else:
            if self.verbosity >= 2:
                print(f"{self.dump_directory}/{filename}: new text image file")
            self.load_file(filename)
            new_file = True

        # Back up lo-res and save hi-res file
        self.backup_file(filename)
        if filename in self.assigned_files:
            self.save_hires_file(filename)

        return new_file

    def save_hdf5_cache(self) -> None:
        """
        Saves character and text image file data structures to hdf5 cache
        """
        if self.verbosity >= 1:
            print(f"Saving cache to '{self.cache_file}'")
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
        if (isfile(f"{self.load_directory}/{filename}")
                and not self.operations["overwrite"]):
            return
        match = self.re_filename.match(filename)
        if match is None:
            raise ValueError()
        line_width = int(match["width"]) // 16
        text_height = int(match["height"]) // 16

        text_data = np.zeros((text_height * self.size,
                              line_width * self.size, 4), np.uint8)
        for i, char in enumerate(self.assigned_files[filename]):
            char_data = self.hires_chars[char]
            x = i % line_width
            y = i // line_width
            # @formatter:off
            text_data[y * self.size:(y + 1) * self.size,
                      x * self.size:(x + 1) * self.size, 3] = char_data
            # @formatter:on

        Image.fromarray(text_data).save(f"{self.load_directory}/{filename}")
        if self.verbosity >= 1:
            print(f"{self.load_directory}/{filename} saved")

    def save_load_directory(self) -> None:
        """
        Saves high resolution images to Citra's load directory
        """
        if self.verbosity >= 1:
            print(f"Saving images to '{self.load_directory}'")
        for filename in self.assigned_files:
            self.save_hires_file(filename)

    def scan_dump_directory(self) -> int:
        """
        Scans dump directory for files

        Returns:
            int: number of new files
        """
        n_files = 0
        if self.dump_directory is None:
            raise ValueError

        if self.verbosity >= 1:
            print(f"Scanning images in '{self.dump_directory}'")
        for filename in listdir(self.dump_directory):
            n_files += self.process_file(filename)

        return n_files

    def watch_dump_directory(self) -> int:
        """
        Watches dump directory for new files

        Returns:
            int: number of new files observed
        """
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as e:
            raise e

        class FileCreatedEventHandler(FileSystemEventHandler):  # type: ignore
            """
            Handles file creation in dump folder
            """

            def __init__(self, host) -> None:  # type: ignore
                """
                Initializes

                Args:
                    host (OOT3DHDTextGenerator): Host to which files will be
                       passed
                """
                self.host = host
                self.n_new_images = 0

            def on_created(self, event):  # type: ignore
                """
                Handles a file creation event

                Args:
                    event: File creation event whose file to process
                """
                filename = basename(event.key[1])
                self.n_new_images += self.host.process_file(filename)

        if self.dump_directory is None:
            raise ValueError()

        if self.verbosity >= 1:
            print(f"Watching for new images in '{self.dump_directory}'")
        event_handler = FileCreatedEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.dump_directory)
        observer.start()
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

        return event_handler.n_new_images

    # endregion

    # region Class Methods

    @classmethod
    def construct_argparser(cls) -> ArgumentParser:
        """
        Constructs argument parser

        Returns:
            parser (ArgumentParser): Argument parser
        """

        def infile_argument(value: str) -> str:
            if not isinstance(value, str):
                raise ArgumentError()

            value = expandvars(value)
            if not isfile(value):
                raise ArgumentError(f"infile '{value}' does not exist")
            elif not access(value, R_OK):
                raise ArgumentError(f"infile '{value}' cannot be read")

            return value

        parser = ArgumentParser(
            description=__doc__,
            formatter_class=RawDescriptionHelpFormatter)
        verbosity = parser.add_mutually_exclusive_group()
        verbosity.add_argument(
            "-v", "--verbose",
            action="count",
            default=1,
            dest="verbosity",
            help="enable verbose output, may be specified more than once")
        verbosity.add_argument(
            "-q", "--quiet",
            action="store_const",
            const=0,
            dest="verbosity",
            help="disable verbose output")
        parser.add_argument(
            "conf_file",
            type=infile_argument,
            help="configuration file")

        return parser

    @classmethod
    def main(cls) -> None:
        """Parses and validates arguments, constructs and calls object"""

        parser = cls.construct_argparser()
        kwargs = vars(parser.parse_args())
        cls(**kwargs)()

    # endregion

    # region Static Methods

    @staticmethod
    def concatenate_images(*images: Image.Image) -> Image.Image:
        """
        Horizontally concatenates a series of images

        Args:
            *images (Image.Image): two or more images to concatenate

        Returns:
            Image.Image: concatenated image
        """
        width = 0
        height = 0
        for image in images:
            width += image.size[0]
            height = max(height, image.size[1])

        concatenated_image = Image.new("RGB", (width, height))
        x = 0
        for image in images:
            concatenated_image.paste(image, (x, 0))
            x += image.size[0]

        return concatenated_image

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
    OOT3DHDTextGenerator.main()
