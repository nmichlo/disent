import torch
import numpy as np
import time

"""
helpful functions that do not fit nicely into any other file.
"""

# ========================================================================= #
# seeds                                                                     #
# ========================================================================= #


def seed(long=777):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(long)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(long)
    print(f'[SEEDED]: {long}')


# ========================================================================= #
# IO                                                                        #
# ========================================================================= #


def to_numpy(array):
    """
    Handles converting any array like object to a numpy array.
    specifically with support for a tensor
    """
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    # TODO: is this really necessary? recursion?
    # elif isinstance(array, (list, tuple)):
    #     return np.array([to_numpy(elem for elem in array)])
    else:
        return np.array(array)


# ========================================================================= #
# IO                                                                        #
# ========================================================================= #


def atomic_save(obj, path):
    """
    Save a model to a file, making sure that the file will
    never be partly written.

    This prevents the model from getting corrupted in the
    event that the process dies or the machine crashes.

    FROM: my obstacle_tower project
    """
    import os
    import torch

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + '.tmp')
    os.rename(path + '.tmp', path)

def save_model(model, path):
    atomic_save(model.state_dict(), path)
    print(f'[MODEL]: saved {path}')

def load_model(model, path, cuda=True, skip_if_missing=False):
    """
    FROM: my obstacle_tower project
    """

    import os
    import torch

    if path and os.path.exists(path):
        print(f'[MODEL]: loaded {path} (cuda: {cuda})')
        model.load_state_dict(torch.load(
            path,
            map_location=torch.device('cuda' if cuda else 'cpu')
        ))
    elif skip_if_missing:
        raise Exception(f'Could not initialise the model from: {path}')
    if cuda:
        model = model.to(torch.device('cuda'))  # this needs to stay despite the above.
        print('[MODEL]: Moved to GPU')
    return model


# ========================================================================= #
# STRINGS                                                                   #
# ========================================================================= #


def print_separator(text, header=None, width=100, char_v='#', char_h='=', char_corners=None):
    """
    function wraps text between two lines or inside a box with lines on either side.
    FROM: my obstacle_tower project
    """
    if char_corners is None:
        char_corners = char_v
    assert len(char_v) == len(char_corners)
    assert len(char_h) == 1
    import textwrap
    import pprint
    from tqdm import tqdm

    def append_wrapped(text):
        for line in text.splitlines():
            for wrapped in (textwrap.wrap(line, w, tabsize=4) if line.strip() else ['']):
                lines.append(f'{char_v} {wrapped:{w}s} {char_v}')

    w = width-4
    lines = []
    sep = f'{char_corners} {char_h*w} {char_corners}'
    lines.append(f'\n{sep}')
    if header:
        append_wrapped(header)
        lines.append(sep)
    if type(text) != str:
        text = pprint.pformat(text, width=w)
    append_wrapped(text)
    lines.append(f'{sep}\n')
    tqdm.write('\n'.join(lines))

def print_box(text, header=None, width=100):
    """
    like print_separator but is isntead a box
    FROM: my obstacle_tower project
    """
    print_separator(text, header=header, width=width, char_v='|', char_h='-', char_corners='#')


# ========================================================================= #
# Timing                                                                    #
# ========================================================================= #


# class Measure:
#     """
#     This class is intended to measure the runtime of a section of code, using the 'with' statement.
#     eg.
#
#     with Measure("My Timer"):
#         print("I am being timed")
#
#     TODO: change so that this averages all the iterations within one level of the call stack.
#
#     FROM: my obstacle_tower project
#     """
#
#     _call_stack = []
#     _default_printer = print
#
#     def __init__(self, name, printer=_default_printer, verbose=False):
#         assert name and printer
#         self.start = None
#         self.end = None
#         self.name = name
#         self.printer = printer
#         self._verbose = verbose
#
#     def _print(self, *text):
#         name = '/'.join(Measure._call_stack)
#         text = ''.join(str(t) for t in text)
#         self.printer(f"[\033[93m{name}\033[0m] {text}")
#
#     def __enter__(self):
#         Measure._call_stack.append(self.name)
#         if self._verbose:
#             self._print("\033[92m", "Started", "\033[0m")
#         self.start = time.time()
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.end = time.time()
#         if self._verbose:
#             self._print("\033[91m", "Finished", "\033[0m", ": ", self.duration(), "ms")
#         else:
#             self._print("\033[91m", "Done", "\033[0m", ": ", self.duration(), "ms")
#         Measure._call_stack.pop()
#
#     def duration(self):
#         return round((self.end - self.start) * 1000 * 1000) / 1000
#
#     @staticmethod
#     def timer_builder(printer=_default_printer):
#         return lambda name: Measure(name, printer=printer)


# ========================================================================= #
# Config Files                                                              #
# ========================================================================= #


# def config_file(config_path, default=None):
#     """
#     Decorator that converts a class to a named tuple,
#     setting or replacing defaults from a config file.
#     TODO: Maybe i should remove this....
#
#     FROM: my obstacle_tower project
#     """
#
#     def wrapper(cls):
#         from typing import get_type_hints
#         from collections import namedtuple
#         import toml
#         import os
#         # load fields
#         fields = get_type_hints(cls)
#         field_defaults = {k: getattr(cls, k) for k in fields if hasattr(cls, k)}
#         field_values = field_defaults
#         # load config if file is spcified
#         if os.path.isfile(config_path):
#             config_values = toml.load(config_path)
#             if default:
#                 assert default in config_values, f'Config defaults "{default}" does not exist for "{cls.__name__}" in "{config_path}"'
#                 config_values = config_values[default]
#             field_values = {**field_values, **config_values}
#         # verify
#         extra = set(field_values) - set(fields)
#         uninitialised = set(fields) - set(field_values)
#         if extra or uninitialised:
#             print(
#                 f'Config: {cls.__name__} fields mismatch\n'
#                 f' - Invalid fields: {extra}\n'
#                 f' - Uninitialised fields: {uninitialised}\n'
#             )
#             raise KeyError('Fields mismatch')
#         # check types
#         for k, t in fields.items():
#             try:
#                 field_values[k] = t(field_values[k])
#             except:
#                 assert False, f'"{k}" on "{cls.__name__}" cannot be cast to: {t}'
#         # Return named tuple instance
#         return namedtuple(cls.__name__, list(fields))(**field_values)
#     return wrapper


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
