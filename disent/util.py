import gin
import torch
import numpy as np


"""
helpful functions that do not fit nicely into any other file.
"""


# ========================================================================= #
# logging                                                                   #
# ========================================================================= #


IS_LOGGING_INITIALISED = False


@gin.configurable('log', whitelist=['level'])
def make_logger(name=None, level='INFO'):
    """
    function that returns a logger with defaults.
    automatically retrieves the module name from the stack.
    """
    global IS_LOGGING_INITIALISED
    # initialise logging once
    if not IS_LOGGING_INITIALISED:
        import coloredlogs
        coloredlogs.install(
            level=level,
            fmt='[%(asctime)s][%(hostname)s][%(name)s:%(process)d][%(levelname)s]: %(message)s',
            field_styles=coloredlogs.parse_encoded_styles('asctime=white;hostname=cyan;name=magenta;process=red;levelname=yellow'),
            level_styles=coloredlogs.parse_encoded_styles('debug=green;info=white;warning=yellow;error=red;critical=bold,background=red'),
        )
        IS_LOGGING_INITIALISED = True
    # get name if needed from stack | 2 if using gin.config otherwise 1
    if name is None:
        import inspect
        name = inspect.getmodule(inspect.stack()[2].frame).__name__
    # make logger
    import logging
    return logging.getLogger(name)


log = make_logger()


# ========================================================================= #
# seeds                                                                     #
# ========================================================================= #


def seed(long=777):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    # TODO: this is automatically handled by the sacred experiment manager if we transition to that.
    #       just check... except for torch.backends?
    torch.manual_seed(long)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(long)
    log.info(f'[SEEDED]: {long}')


class TempNumpySeed(object):
    def __init__(self, seed=None, offset=0):
        if seed is not None:
            try:
                seed = int(seed)
            except:
                raise ValueError(f'{seed=} is not int-like!')
        self._seed = seed
        if seed is not None:
            self._seed += offset
        self._state = None

    def __enter__(self):
        if self._seed is not None:
            self._state = np.random.get_state()
            np.random.seed(self._seed)

    def __exit__(self, *args, **kwargs):
        if self._seed is not None:
            np.random.set_state(self._state)
            self._state = None

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
    # recursive conversion
    # not super efficient but allows handling of PIL.Image and other nested data.
    elif isinstance(array, (list, tuple)):
        return np.array([to_numpy(elem) for elem in array])
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
    log.info(f'[MODEL]: saved {path}')

def load_model(model, path, cuda=True, fail_if_missing=True):
    """
    FROM: my obstacle_tower project
    """
    import os
    import torch

    if path and os.path.exists(path):
        model.load_state_dict(torch.load(
            path,
            map_location=torch.device('cuda' if cuda else 'cpu')
        ))
        log.info(f'[MODEL]: loaded {path} (cuda: {cuda})')
    else:
        if fail_if_missing:
            raise Exception(f'Could not load model, path does not exist: {path}')
    if cuda:
        model = model.cuda()  # this needs to stay despite the above.
        log.info('[MODEL]: Moved to GPU')
    return model


# ========================================================================= #
# Iterators                                                                 #
# ========================================================================= #


def chunked(arr, chunk_size=1):
    for i in range(len(arr) // chunk_size):
        yield arr[chunk_size*i:chunk_size*(i+1)]


# ========================================================================= #
# STRINGS                                                                   #
# ========================================================================= #

# TODO: make this return a string not actually print out so it can be used with logging
def make_separator_str(text, header=None, width=100, char_v='#', char_h='=', char_corners=None):
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
    return '\n'.join(lines)

def make_box_str(text, header=None, width=100, char_v='|', char_h='-', char_corners='#'):
    """
    like print_separator but is isntead a box
    FROM: my obstacle_tower project
    """
    return make_separator_str(text, header=header, width=width, char_v=char_v, char_h=char_h, char_corners=char_corners)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
