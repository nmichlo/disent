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


# ========================================================================= #
# Working With Arbitrary Precision Integers                                 #
# ========================================================================= #


def gcd(a: int, b: int) -> int:
    """
    Compute the greatest common divisor of a and b
    TODO: not actually sure if this returns the correct values for zero or negative inputs?
    """
    assert isinstance(a, int), f'number must be an int, got: {type(a)}'
    assert isinstance(b, int), f'number must be an int, got: {type(b)}'
    while b > 0:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    Compute the lowest common multiple of a and b
    TODO: not actually sure if this returns the correct values for zero or negative inputs?
    """
    return (a * b) // gcd(a, b)


# ========================================================================= #
# End                                                                       #
# ========================================================================= #
