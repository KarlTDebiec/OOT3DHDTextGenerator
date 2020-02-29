#!python
# -*- coding: utf-8 -*-
#   Upscaler.py
#
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license.
################################### MODULES ###################################
from __future__ import annotations

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
from typing import Dict, List, Optional, Any, Set

import h5py
import numpy as np
import yaml
from IPython import embed
from PIL import Image, ImageChops, ImageDraw, ImageFont, UnidentifiedImageError


################################### CLASSES ###################################
class Upscaler:
    """
    TODO:
        - Test separately tracing black stroke and white fill
    """

    # region Class Variables

    package_root: str = str(Path(__file__).parent.absolute())

    # endregion

    # region Builtins

    def __init__(self, conf_file: str = "conf_upscaler.yaml") -> None:
        """
        Initializes

        Args:
            conf_file (str): file from which to load configuration
        """
        # Read configuration file
        conf_file = expandvars(conf_file)
        if not (isfile(conf_file) and access(conf_file, R_OK)):
            raise ValueError(f"Configuration file '{conf_file}' could not be "
                             f"read")
        with open(conf_file, "r") as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)

        # General configuration
        self.verbosity = conf.get("verbosity", 1)

        # Input configuration
        self.input_directory = conf.get("input", getcwd())

        # Output configuration
        self.output_directory = conf.get("output", getcwd())

    def __call__(self) -> None:
        # infile = "卡卡利科村.png"
        # prefix = ".".join(infile.split(".")[:-1])
        # created_files: Set[str] = set()

        # Make background white
        # flat_file = f"{prefix}_flat.bmp"
        # if not isfile(flat_file):
        #     self.flatten(infile, flat_file)
        # flat_image = Image.open(flat_file).convert("RGB")

        # Scale image
        # imagetype = "a"
        # scale = 2
        # noise = 4
        # scale_params = f"waifu-{imagetype}-{scale}-{noise}"
        # scale_file = f"{prefix}_{scale_params}.bmp"
        # scale_params = f"pixelmator-9x-sharpen"
        # scale_file = f"{prefix}_{scale_params}.bmp"
        # if not isfile(scale_file):
        #     print(scale_file)
        #     self.scale_pixelmator(flat_file, scale_file)

        # Trace image
        # blacklevels = np.arange(0.05, 0.30, 0.05)
        # alphamaxes = np.arange(1, 6, 1)
        # opttolerances = np.arange(0.2, 1.2, 0.2)
        # best_params = 0
        # best_diff = 1000000
        # best_files: Set[str] = set()
        # for blacklevel in blacklevels:
        #     for alphamax in alphamaxes:
        #         for opttolerance in opttolerances:
        #             trace_params = f"potrace-{blacklevel:3.2f}-{alphamax}-" \
        #                            f"{opttolerance:3.1f}"
        #             vector_file = f"{prefix}_{scale_params}_{trace_params}.svg"
        #             raster_file = f"{prefix}_{scale_params}_{trace_params}.png"
        #             if not isfile(raster_file):
        #                 if not isfile(vector_file):
        #                     self.trace_potrace(scale_file, vector_file,
        #                                        blacklevel, alphamax,
        #                                        opttolerance)
        #                 self.convert(vector_file, raster_file)
        #             created_files.add(vector_file)
        #             created_files.add(raster_file)
        #
        #             shrunk_file = f"{prefix}_{scale_params}_{trace_params}_" \
        #                           f"shrunk.png"
        #             shrunk_image = Image.open(
        #                 raster_file).convert("RGB").resize(
        #                 (256, 32), Image.BILINEAR)
        #             shrunk_image.save(shrunk_file)
        #             created_files.add(shrunk_file)
        #
        #             diff_file = f"{prefix}_{scale_params}_{trace_params}_" \
        #                         f"diff.png"
        #             diff_image = ImageChops.difference(flat_image,
        #                                                shrunk_image)
        #             diff_image.save(diff_file)
        #             created_files.add(diff_file)
        #
        #             if np.array(diff_image).sum() < best_diff:
        #                 best_params = trace_params
        #                 best_diff = np.array(diff_image).sum()
        #                 best_files: Set[str] = {vector_file, raster_file,
        #                                         shrunk_file, diff_file}
        #
        #             if self.verbosity >= 1:
        #                 print(f"{scale_params} {trace_params} "
        #                       f"{np.array(diff_image).sum()}")
        #                 print()
        # print(f"{best_params} {best_diff}")
        # for suboptimal_file in created_files - best_files:
        #     remove(suboptimal_file)

    # endregion

    # region Properties

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

    # endregion

    # region Methods

    def convert(self, infile: str, outfile: str) -> None:
        if self.verbosity >= 1:
            print(f"Converting to '{outfile}'")
        command = f"convert " \
                  f"{infile} " \
                  f"{outfile}"
        Popen(command, shell=True, stdin=PIPE, stdout=DEVNULL,
              stderr=DEVNULL, close_fds=True).wait()

    def flatten(self, infile: str, outfile: str) -> None:
        if self.verbosity >= 1:
            print(f"Flattening transparency to '{outfile}'")
        raw_data = np.array(Image.open(infile))
        flat_data = np.ones_like(raw_data) * 255
        flat_data[:, :, 0] = 255 - raw_data[:, :, 3]
        flat_data[:, :, 1] = 255 - raw_data[:, :, 3]
        flat_data[:, :, 2] = 255 - raw_data[:, :, 3]
        flat_data[:, :, :3] += raw_data[:, :, :3]
        Image.fromarray(flat_data).convert("RGB").save(outfile)

    def trace_potrace(self, infile: str, outfile: str, blacklevel: float,
                      alphamax: int, opttolerance: float) -> None:
        if self.verbosity >= 1:
            print(f"Tracing to '{outfile}'")
        command = f"potrace {infile} " \
                  f"-b svg " \
                  f"-k {blacklevel} " \
                  f"-a {alphamax} " \
                  f"-O {opttolerance} " \
                  f"-o {outfile}"
        Popen(command, shell=True, stdin=PIPE, stdout=DEVNULL,
              stderr=DEVNULL, close_fds=True).wait()

    # endregion

    # region Scaling Methods

    def scale_pixelmator(self, infile: str, outfile: str) -> None:
        if self.verbosity >= 1:
            print(f"Scaling to '{outfile}'")
        copyfile(infile, outfile)
        command = f"/usr/bin/automator " \
                  f"-i {outfile} " \
                  f"{self.package_root}/Pixelmator\ Scale9x\ Sharpen.workflow"
        Popen(command, shell=True, close_fds=True).wait()

    def scale_waifu(self, infile: str, outfile: str, imagetype: str = "a",
                    scale: int = 2, noise: int = 4) -> None:
        # Need to scale first, waifu will not work on very small images
        command = f"/Users/kdebiec/OneDrive/code/external/waifu2x/waifu2x " \
                  f"-t {imagetype} " \
                  f"-s {scale} " \
                  f"-n {noise} " \
                  f"-i {infile} " \
                  f"-o {outfile}"
        Popen(command, shell=True, stdin=PIPE, stdout=DEVNULL,
              stderr=DEVNULL, close_fds=True).wait()

    def scale_xbrz(self, infile: str, outfile, str):
        pass

    # endregion

    # region Static Methods

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
    Upscaler()()
