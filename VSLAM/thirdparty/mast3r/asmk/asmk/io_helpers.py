"""Helper functions related to io"""

import os.path
import time
import sys
import logging
import pickle
import urllib.request
from io import StringIO
from pathlib import Path
import yaml
import numpy as np

# Params

def load_params(path):
    """Return loaded parameters from a yaml file"""
    with open(path, "r") as handle:
        content = yaml.safe_load(handle)
    if "__template__" in content:
        # Treat template as defaults
        template_path = os.path.expanduser(content.pop("__template__"))
        template = load_params(os.path.join(os.path.dirname(path), template_path))
        content = dict_deep_overlay(template, content)
    return content

def dict_deep_overlay(defaults, params):
    """If defaults and params are both dictionaries, perform deep overlay (use params value for
        keys defined in params, otherwise use defaults value)"""
    if isinstance(defaults, dict) and isinstance(params, dict):
        for key in params:
            defaults[key] = dict_deep_overlay(defaults.get(key, None), params[key])
        return defaults

    return params


# Logging

def init_logger(log_path):
    """Return a logger instance which logs to stdout and, if log_path is not None, also to a file"""
    logger = logging.getLogger("ASMK")
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
    logger.addHandler(stdout_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def progress(iterable, *, size=None, frequency=1, header=""):
    """Generator that wraps an iterable and prints progress"""
    if size is None:
        size = len(iterable)
    header = f"{header.capitalize()}: " if header else ""
    charsize = len(str(size))
    if frequency:
        print(f"{header}[{'0'.rjust(charsize)}/{size}]", end="  ")
        sys.stdout.flush()
    time0 = time.time()
    for i, element in enumerate(iterable):
        yield element
        i1 = i+1
        if frequency and (i1 % frequency == 0 or i1 == size):
            avg_time = (time.time() - time0) / i1
            print(f"\r{header}[{str(i1).rjust(charsize)}/{size}] " \
                    f"elapsed {int(avg_time*i1/60):02d}m/{int(avg_time*size/60):02d}m", end="  ")
            sys.stdout.flush()
    if frequency:
        # print()
        pass

def capture_stdout(func, logger):
    """Redirect stdout to logger"""
    sys.stdout, stdout = StringIO(), sys.stdout
    func()
    sys.stdout, out_text = stdout, sys.stdout.getvalue()
    for line in out_text.strip().split("\n"):
        logger.info(line)


# Load and save state dicts

def load_pickle(path):
    """Load pickled data from path"""
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(path, data):
    """Save data to path using pickle"""
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


# Download

def download_files(names, root_path, base_url, logfunc=None):
    """Download file names from given url to given directory path. If logfunc given, use it to log
        status."""
    root_path = Path(root_path)
    for name in names:
        path = root_path / name
        if path.exists():
            continue
        if logfunc:
            logfunc(f"Downloading file '{name}'")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base_url + name, path)


# Iteration

def slice_unique(ids):
    """Generate slices that mark a sequence of identical values in a given array of ids. The
        sequence must be uninterrupted (compact)."""
    pointer = 0
    for i, counts in zip(*np.unique(ids, return_counts=True)):
        seq = slice(pointer, pointer+counts)
        assert (ids[seq] == i).all()
        yield i, seq
        pointer += counts
