#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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

import os
import re
import torch
import research


# TODO: register these kernels to disent!
_ARG_KERNELS = [
    # (REGEX, EXAMPLE, FACTORY_FUNC)
    # - factory function takes at min one arg: fn(reduction) with one arg after that per regex capture group
    # - regex expressions are tested in order, expressions should be mutually exclusive or ordered such that more specialized versions occur first.
    (re.compile(r'^(xy8)_r(47)$'),  'xy8_r47', lambda kern, radius: torch.load(os.path.abspath(os.path.join(research.__file__, '../part03_adversarial/e01_learn_to_disentangle/data', 'r47-1_s28800_adam_lr0.003_wd0.0_xy8x8.pt')))),
    (re.compile(r'^(xy1)_r(47)$'),  'xy1_r47', lambda kern, radius: torch.load(os.path.abspath(os.path.join(research.__file__, '../part03_adversarial/e01_learn_to_disentangle/data', 'r47-1_s28800_adam_lr0.003_wd0.0_xy1x1.pt')))),
]
