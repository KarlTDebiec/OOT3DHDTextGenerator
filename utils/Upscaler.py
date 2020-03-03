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

from abc import ABC, abstractmethod
from collections import OrderedDict
from os import R_OK, W_OK, access, getcwd, listdir, makedirs, remove
from os.path import basename, dirname, expandvars, isdir, isfile, join, \
    splitext
from pathlib import Path
from shutil import copyfile, which
from subprocess import Popen
from sys import modules
from typing import Any, Generator, List, Optional, Union, no_type_check

import numba as nb
import numpy as np
import yaml
from PIL import Image


################################### CLASSES ###################################
class Processor(ABC):
    executable_name: Optional[str] = None
    extension: str = "png"

    def __init__(self, **kwargs: Any) -> None:
        self.paramstring = kwargs.get("paramstring", None)
        if self.executable_name is not None:
            self.executable = expandvars(
                str(kwargs.get("executable", which(self.executable_name))))

    def __call__(self, downstream_processors: Optional[List[
        Generator[None, str, None]]] = None) -> Generator[None, str, None]:
        while True:
            infile = (yield)
            outfile = self.get_outfile(infile)
            self.process_file(infile, outfile)
            if downstream_processors is not None:
                for downstream_processor in downstream_processors:
                    downstream_processor.send(outfile)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.paramstring}>"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.paramstring}>"

    def get_outfile(self, infile: str) -> str:
        outfile = splitext(basename(infile))[0].lstrip("original")
        if self.paramstring != "":
            outfile += f"_{self.paramstring}"
        outfile = outfile.lstrip("_")
        outfile += f".{self.extension}"
        return f"{dirname(infile)}/{outfile}"

    @abstractmethod
    def process_file(self, infile: str, outfile: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        pass


class CopyProcessor(Processor):
    def __init__(self, output_directory: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_directory = expandvars(output_directory)

    def get_outfile(self, infile: str) -> str:
        return f"{self.output_directory}/{basename(dirname(infile))}.png"

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        copyfile(infile, outfile)

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        output_directories = kwargs.pop("output_directory")
        if not isinstance(output_directories, list):
            output_directories = [output_directories]

        processors: List[Processor] = []
        for output_directory in output_directories:
            processors.append(cls(
                output_directory=output_directory,
                **kwargs))
        return processors


class Flattener(Processor):
    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Flattening to '{outfile}'")
        input_data = np.array(Image.open(infile))
        output_data = np.ones_like(input_data) * 255
        output_data[:, :, 0] = 255 - input_data[:, :, 3]
        output_data[:, :, 1] = 255 - input_data[:, :, 3]
        output_data[:, :, 2] = 255 - input_data[:, :, 3]
        output_data[:, :, :3] += input_data[:, :, :3]
        Image.fromarray(output_data).convert("RGB").save(outfile)

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        return [cls(paramstring=f"flatten")]


class ImageMagickProcessor(Processor):
    executable_name = "convert"

    def __init__(self, extension: str, resize: Any = False,
                 remove_infile: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.extension = extension
        self.remove_infile = remove_infile
        self.resize = resize

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Processing to '{outfile}'")
        if self.resize:
            command = f"{self.executable} " \
                      f"-resize {self.resize[0]}x{self.resize[1]} " \
                      f"{infile} " \
                      f"{outfile}"
        else:
            command = f"{self.executable} " \
                      f"{infile} " \
                      f"{outfile}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()
        if self.remove_infile:
            remove(infile)

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        return [cls(extension=kwargs.pop("extension", "bmp"),
                    remove_infile=kwargs.pop("remove_infile", False),
                    resize=kwargs.pop("resize", False),
                    paramstring="",
                    **kwargs)]


class PixelmatorProcessor(Processor):
    executable_name = "automator"

    def __init__(self, workflow: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.workflow = workflow

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Processing to '{outfile}'")
        copyfile(infile, outfile)
        command = f"{self.executable} " \
                  f"-i {outfile} " \
                  f"{self.workflow}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        if "workflow_directory" in kwargs:
            workflow_directory = expandvars(
                str(kwargs.pop("workflow_directory")))
        else:
            workflow_directory = f"{Path(__file__).parent.absolute()}/" \
                                 f"workflows"
        workflows: Union[List[str], str] = kwargs.pop("workflow")
        if not isinstance(workflows, list):
            workflows = [workflows]

        processors: List[Processor] = []
        for workflow in workflows:
            processors.append(cls(
                workflow=f"{workflow_directory}/{workflow}.workflow",
                paramstring=f"pixelmator-"
                            f"{workflow}",
                **kwargs))
        return processors


class PotraceProcessor(Processor):
    executable_name = "potrace"

    def __init__(self, blacklevel: float, alphamax: float, opttolerance: float,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.blacklevel = blacklevel
        self.alphamax = alphamax
        self.opttolerance = opttolerance

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Tracing to '{outfile}'")

        # Convert to bmp; potrace does not accept png
        bmpfile = f"{splitext(infile)[0]}.bmp"
        command = f"convert " \
                  f"{infile} " \
                  f"{bmpfile}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()

        # trace
        svgfile = f"{splitext(outfile)[0]}.svg"
        command = f"{self.executable} " \
                  f"{bmpfile} " \
                  f"-b svg " \
                  f"-k {self.blacklevel} " \
                  f"-a {self.alphamax} " \
                  f"-O {self.opttolerance} " \
                  f"-o {svgfile}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()

        # Rasterize svg to png
        pngfile = f"{splitext(outfile)[0]}.png"
        command = f"convert " \
                  f"{svgfile} " \
                  f"{pngfile}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()

        remove(bmpfile)
        remove(svgfile)

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        blacklevels = kwargs.pop("blacklevel")
        if not isinstance(blacklevels, list):
            blacklevels = [blacklevels]
        alphamaxes = kwargs.pop("alphamax")
        if not isinstance(alphamaxes, list):
            alphamaxes = [alphamaxes]
        opttolerances = kwargs.pop("opttolerance")
        if not isinstance(opttolerances, list):
            opttolerances = [opttolerances]

        processors: List[Processor] = []
        for blacklevel in blacklevels:
            for alphamax in alphamaxes:
                for opttolerance in opttolerances:
                    processors.append(cls(
                        blacklevel=blacklevel,
                        alphamax=alphamax,
                        opttolerance=opttolerance,
                        paramstring=f"potrace-"
                                    f"{float(blacklevel):3.2f}-"
                                    f"{alphamax}-"
                                    f"{float(opttolerance):3.1f}",
                        **kwargs))
        return processors


class ResizeProcessor(Processor):

    def __init__(self, scale: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Processeing to '{outfile}'")
        input_image = Image.open(infile).convert("L")
        output_image = input_image.resize((
            int(np.round(input_image.size[0] * self.scale)),
            int(np.round(input_image.size[1] * self.scale))))
        output_image.save(outfile)

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        scales = kwargs.pop("scale")
        if not isinstance(scales, list):
            scales = [scales]

        processors: List[Processor] = []
        for scale in scales:
            processors.append(cls(
                scale=scale,
                paramstring=f"resize-{scale:7.5f}",
                **kwargs))
        return processors


class ThresholdProcessor(Processor):

    def __init__(self, threshold: int = 128, denoise: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.denoise = denoise

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Processeing to '{outfile}'")
        input_image = Image.open(infile).convert("L").point(
            lambda p: p > self.threshold and 255)
        data = np.array(input_image)
        if self.denoise:
            self.denoise_data(data)
        processed_image = Image.fromarray(data)
        processed_image.save(outfile)

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        thresholds = kwargs.pop("threshold")
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        denoises = kwargs.pop("denoise")
        if not isinstance(denoises, list):
            denoises = [denoises]

        processors: List[Processor] = []
        for threshold in thresholds:
            for denoise in denoises:
                if denoise:
                    paramstring = f"threshold-{threshold}-denoise"
                else:
                    paramstring = f"threshold-{threshold}"
                processors.append(cls(
                    threshold=threshold,
                    denoise=denoise,
                    paramstring=paramstring,
                    **kwargs))
        return processors

    @no_type_check
    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
    def denoise_data(data: np.ndarray) -> None:
        for x in range(1, data.shape[1] - 1):
            for y in range(1, data.shape[0] - 1):
                slc = data[y - 1:y + 2, x - 1:x + 2]
                if data[y, x] == 0:
                    if (slc == 0).sum() < 4:
                        data[y, x] = 255
                else:
                    if (slc == 255).sum() < 4:
                        data[y, x] = 0


class WaifuProcessor(Processor):
    executable_name = "waifu2x"

    def __init__(self, imagetype: str, scale: str, noise: str,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.imagetype = imagetype
        self.scale = scale
        self.noise = noise

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Processing to '{outfile}'")
        command = f"{self.executable} " \
                  f"-t {self.imagetype} " \
                  f"-s {self.scale} " \
                  f"-n {self.noise} " \
                  f"-i {infile} " \
                  f"-o {outfile}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        processors: List[Processor] = []
        imagetypes = kwargs.pop("imagetype")
        scales = kwargs.pop("scale")
        noises = kwargs.pop("noise")
        for imagetype in imagetypes:
            for scale in scales:
                for noise in noises:
                    processors.append(cls(
                        imagetype=imagetype,
                        scale=scale,
                        noise=noise,
                        paramstring=f"waifu-"
                                    f"{imagetype}-"
                                    f"{scale}-"
                                    f"{noise}",
                        **kwargs))
        return processors


class XbrzProcessor(Processor):
    executable_name = "xbrzscale"

    def __init__(self, scale: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def process_file(self, infile: str, outfile: str) -> None:
        if isfile(outfile):
            return
        print(f"Processing to '{outfile}'")
        command = f"{self.executable} " \
                  f"{self.scale} " \
                  f"{infile} " \
                  f"{outfile}"
        print(command)
        Popen(command, shell=True, close_fds=True).wait()

    @classmethod
    def get_processors(cls, **kwargs: Any) -> List[Processor]:
        processors: List[Processor] = []
        scales = kwargs.pop("scale")
        for scale in scales:
            processors.append(cls(
                scale=scale,
                paramstring=f"xbrz-"
                            f"{scale}",
                **kwargs))
        return processors


class Upscaler:
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

        # Preprocessor configuration
        self.pipeline: OrderedDict[str, List[Processor]] = OrderedDict()
        for stage in conf["pipeline"]:
            stage_name = list(stage.keys())[0]
            processors: List[Processor] = []
            for processor in list(stage.values())[0]:
                if isinstance(processor, dict):
                    processor_name = list(processor.keys())[0]
                    processor_args = list(processor.values())[0]
                else:
                    processor_name = processor
                    processor_args = {}
                processor_cls = getattr(modules[__name__], processor_name)
                processors.extend(
                    processor_cls.get_processors(**processor_args))
            self.pipeline[stage_name] = processors

        # Output configuration
        self.output_directory = conf.get("output", getcwd())

    def __call__(self) -> None:
        """
        Performs operations
        """
        downstream_processors = None
        for stage in reversed(self.pipeline.keys()):
            print(stage)
            processors = []
            for processor_object in self.pipeline[stage]:
                processor_generator = processor_object(downstream_processors)
                next(processor_generator)
                processors.append(processor_generator)
            downstream_processors = processors

        self.scan_input_directory(downstream_processors)

    # endregion

    # region Properties

    @property
    def input_directory(self) -> Optional[str]:
        """Optional[str]: Directory from which to load lo-res image files"""
        if not hasattr(self, "_input_directory"):
            self._input_directory: Optional[str] = None
        return self._input_directory

    @input_directory.setter
    def input_directory(self, value: Optional[str]) -> None:
        if value is not None:
            value = expandvars(value)
            if not (isdir(value) and access(value, W_OK)):
                raise ValueError()
        self._input_directory = value

    @property
    def output_directory(self) -> Optional[str]:
        """Optional[str]: Directory to which to save hi-res image files"""
        if not hasattr(self, "_output_directory"):
            self._output_directory: Optional[str] = None
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value: Optional[str]) -> None:
        if value is not None:
            value = expandvars(value)
            # TODO: Create if possible
            if not (isdir(value) and access(value, W_OK)):
                raise ValueError()
        self._output_directory = value

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

    def scan_input_directory(self, downstream_processors: Any) -> None:
        print(f"Scanning infiles in '{self.input_directory}'")
        for infile in listdir(self.input_directory):
            if infile == ".DS_Store":
                continue
            infile = join(str(self.input_directory), infile)
            print(f"Processing '{infile}'")
            file_directory = f"{self.output_directory}/" \
                             f"{splitext(basename(infile))[0]}"
            if not isdir(file_directory):
                print(f"Creating directory '{file_directory}'")
                makedirs(file_directory)
            outfile = f"{file_directory}/original.png"
            if not isfile(outfile):
                copyfile(infile, outfile)
                print(f"Copying to '{outfile}'")
            for processor in downstream_processors:
                processor.send(outfile)
            # break

    # endregion


#################################### MAIN #####################################
if __name__ == "__main__":
    Upscaler()()
