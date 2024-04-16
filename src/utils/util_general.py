"""Miscellaneous utility classes and functions."""

import argparse
from pathlib import Path
import os
import numpy as np
import random
import openpyxl
import fnmatch
import torch
import requests
import collections
import sys
import shutil
import ntpath
import re

from typing import Any, Optional, Tuple, Union, List
 
class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()

def print_CUDA_info():
    import torch
    import os
    print("\n")
    print("".center(100, '|'))
    print(" CUDA GPUs REPORT ".center(100, '|'))
    print("".center(100, '|'))
    print("1) Number of GPUs devices: ", torch.cuda.device_count())
    print('2) CUDNN VERSION:', torch.backends.cudnn.version())
    print('3) Nvidia SMI terminal command: \n \n', )
    os.system('nvidia-smi')

    for device in range(torch.cuda.device_count()):
        print("|  DEVICE NUMBER : {%d} |".center(100, '-') % (device))
        print('| 1) CUDA Device Name:', torch.cuda.get_device_name(device))
        print('| 2) CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(device).total_memory / 1e9)
        print("|")

    print("".center(100, '|'))
    print(" GPU REPORT END ".center(100, '|'))
    print("".center(100, '|'))
    print('\n \n')
    return

def list_dict():
   return collections.defaultdict(list)

def nested_dict():
   return collections.defaultdict(nested_dict)

def notification_ifttt(info):
    private_key = "isnY23hWBGyL-mF7F18BUAC-bGAN6dx1UAPoqnfntUa"
    url = "https://maker.ifttt.com/trigger/Notification/json/with/key/" + private_key
    requests.post(url, data={'Info': str(info)})

def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result

def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human-readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])

def seed_all(seed): # for deterministic behaviour
    if not seed:
        seed = 42
    print("Using Seed : ", seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed) # Set torch pseudo-random generator at a fixed value
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed) # Set numpy pseudo-random generator at a fixed value
    random.seed(seed) # Set python built-in pseudo-random generator at a fixed value
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

def parse_separated_list_comma(l):
    if isinstance(l, str):
        return l
    if len(l) == 0:
        return ''
    return ','.join(l)

def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.
    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')
