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

import os
from typing import Dict
from typing import Union


# ========================================================================= #
# file hashing                                                              #
# ========================================================================= #


def _yield_file_bytes(file: str, chunk_size=16384):
    with open(file, 'rb') as f:
        bytes = True
        while bytes:
            bytes = f.read(chunk_size)
            yield bytes


def _yield_fast_hash_bytes(file: str, chunk_size=16384, num_chunks=3):
    assert num_chunks >= 2
    # return the size in bytes
    size = os.path.getsize(file)
    yield size.to_bytes(length=64//8, byteorder='big', signed=False)
    # return file bytes chunks
    if size < chunk_size * num_chunks:
        # we cant return chunks because the file is too small, return everything!
        yield from _yield_file_bytes(file, chunk_size=chunk_size)
    else:
        # includes evenly spaced start, middle and end chunks
        with open(file, 'rb') as f:
            for i in range(num_chunks):
                pos = (i * (size - chunk_size)) // (num_chunks - 1)
                f.seek(pos)
                yield f.read(chunk_size)


# ========================================================================= #
# file hashing                                                              #
# ========================================================================= #


def hash_file(file: str, hash_type='md5', hash_mode='full', missing_ok=True) -> str:
    """
    :param file: the path to the file
    :param hash_type: the kind of hash to compute, default is "md5"
    :param hash_mode: "full" uses all the bytes in the file to compute the hash, "fast" uses the start, middle, end bytes as well as the size of the file in the hash.
    :param chunk_size: number of bytes to read at a time
    :return: the hexdigest of the hash
    :raises FileNotFoundError
    """
    import hashlib
    # check the file exists
    if not os.path.isfile(file):
        if missing_ok:
            return ''
        raise FileNotFoundError(f'could not compute hash for missing file: {repr(file)}')
    # get file bytes iterator
    if hash_mode == 'full':
        byte_iter = _yield_file_bytes(file=file)
    elif hash_mode == 'fast':
        byte_iter = _yield_fast_hash_bytes(file=file)
    else:
        raise KeyError(f'invalid hash_mode: {repr(hash_mode)}')
    # generate hash
    hash = hashlib.new(hash_type)
    for bytes in byte_iter:
        hash.update(bytes)
    hash = hash.hexdigest()
    # done
    return hash


# ========================================================================= #
# file hashing utils                                                        #
# ========================================================================= #


class HashError(Exception):
    """
    Raised if the hash of a file was invalid.
    """


def normalise_hash(hash: Union[str, Dict[str, str]], hash_mode: str) -> str:
    """
    file hashes depend on the mode.
    - Allow hashes to be dictionaries that map the hash_mode to the hash.
      This function returns the correct hash if it is a dictionary.
    """
    if isinstance(hash, dict):
        if hash_mode not in hash:
            raise KeyError(f'hash dictionary does not contain a key for the specified mode: {repr(hash_mode)}, available hashes are: {hash}')
        return hash[hash_mode]
    return hash


def validate_file_hash(file: str, hash: Union[str, Dict[str, str]], hash_type: str = 'md5', hash_mode: str = 'full', missing_ok=True):
    """
    :raises FileNotFoundError, HashError
    """
    hash = normalise_hash(hash=hash, hash_mode=hash_mode)
    # compute the hash
    fhash = hash_file(file=file, hash_type=hash_type, hash_mode=hash_mode, missing_ok=missing_ok)
    # check the hash
    if fhash != hash:
        raise HashError(f'computed {hash_mode} {hash_type} hash: {repr(fhash)} does not match expected hash: {repr(hash)} for file: {repr(file)}')


def is_valid_file_hash(file: str, hash: Union[str, Dict[str, str]], hash_type: str = 'md5', hash_mode: str = 'full', missing_ok=True):
    try:
        validate_file_hash(file=file, hash=hash, hash_type=hash_type, hash_mode=hash_mode, missing_ok=missing_ok)
    except HashError:
        return False
    return True


# ========================================================================= #
# file hashing                                                              #
# ========================================================================= #
