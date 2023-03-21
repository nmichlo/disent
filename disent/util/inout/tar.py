#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2023 Nathan Juraj Michlo
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
from pathlib import Path
from typing import Union


def tar_safe_extract_all(in_file: Union[Path, str], out_dir: Union[Path, str]):
    import tarfile

    in_file = str(in_file)
    out_dir = str(out_dir)

    def _is_safe_to_extract(tar):
        for member in tar.getmembers():
            # check inside directory
            abs_dir = os.path.abspath(out_dir)
            abs_targ = os.path.abspath(os.path.join(out_dir, member.name))
            common_prefix = os.path.commonprefix([abs_dir, abs_targ])
            # raise exception if not
            if common_prefix != abs_dir:
                raise Exception("Attempted path traversal in tar file")

    # this is unsafe tar extraction
    with tarfile.open(in_file) as f:
        _is_safe_to_extract(f)
        f.extractall(out_dir)
