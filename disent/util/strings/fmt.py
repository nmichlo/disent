#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import math
from disent.util.strings import colors as c


# ========================================================================= #
# Byte Formatting                                                           #
# ========================================================================= #


_BYTES_COLR = (c.WHT, c.lGRN, c.lYLW, c.lRED, c.lRED, c.lRED, c.lRED, c.lRED, c.lRED)
_BYTES_NAME = {
    1024: ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"),
    1000: ("B", "KB",  "MB",  "GB",  "TB",  "PB",  "EB",  "ZB",  "YB"),
}


def bytes_to_human(size_bytes: int, decimals: int = 3, color: bool = True, mul: int = 1024) -> str:
    if size_bytes == 0:
        return "0B"
    if mul not in _BYTES_NAME:
        raise ValueError(f'invalid bytes multiplier: {repr(mul)} must be one of: {list(_BYTES_NAME.keys())}')
    # round correctly
    i = int(math.floor(math.log(size_bytes, mul)))
    s = round(size_bytes / math.pow(mul, i), decimals)
    # generate string
    name = f'{_BYTES_COLR[i]}{_BYTES_NAME[mul][i]}{c.RST}' if color else f'{_BYTES_NAME[mul][i]}'
    # format string
    return f"{s:{4+decimals}.{decimals}f} {name}"


# ========================================================================= #
# STRINGS                                                                   #
# ========================================================================= #


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


def concat_lines(*strings, sep=' | '):
    """
    Join multi-line strings together horizontally, with the
    specified separator between them.
    """

    def pad_width(lines):
        max_len = max(len(line) for line in lines)
        return [f'{s:{max_len}}' for s in lines]

    def pad_height(list_of_lines):
        max_lines = max(len(lines) for lines in list_of_lines)
        return [(lines + ([''] * (max_lines - len(lines)))) for lines in list_of_lines]

    list_of_lines = [str(string).splitlines() for string in strings]
    list_of_lines = pad_height(list_of_lines)
    list_of_lines = [pad_width(lines) for lines in list_of_lines]
    return '\n'.join(sep.join(rows) for rows in zip(*list_of_lines))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
