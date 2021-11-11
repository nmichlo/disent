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

from disent.nn.functional._conv2d import torch_conv2d_channel_wise
from disent.nn.functional._conv2d import torch_conv2d_channel_wise_fft

from disent.nn.functional._conv2d_kernels import get_kernel_size
from disent.nn.functional._conv2d_kernels import torch_gaussian_kernel
from disent.nn.functional._conv2d_kernels import torch_gaussian_kernel_2d
from disent.nn.functional._conv2d_kernels import torch_box_kernel
from disent.nn.functional._conv2d_kernels import torch_box_kernel_2d

from disent.nn.functional._correlation import torch_cov_matrix
from disent.nn.functional._correlation import torch_corr_matrix
from disent.nn.functional._correlation import torch_rank_corr_matrix
from disent.nn.functional._correlation import torch_pearsons_corr_matrix
from disent.nn.functional._correlation import torch_spearmans_corr_matrix

from disent.nn.functional._dct import torch_dct
from disent.nn.functional._dct import torch_idct
from disent.nn.functional._dct import torch_dct2
from disent.nn.functional._dct import torch_idct2

from disent.nn.functional._mean import torch_mean_generalized
from disent.nn.functional._mean import torch_mean_quadratic
from disent.nn.functional._mean import torch_mean_geometric
from disent.nn.functional._mean import torch_mean_harmonic

from disent.nn.functional._other import torch_normalize
from disent.nn.functional._other import torch_nan_to_num
from disent.nn.functional._other import torch_unsqueeze_l
from disent.nn.functional._other import torch_unsqueeze_r

from disent.nn.functional._pca import torch_pca_eig
from disent.nn.functional._pca import torch_pca_svd
from disent.nn.functional._pca import torch_pca

# from disent.nn.functional._util_generic import TypeGenericTensor
# from disent.nn.functional._util_generic import TypeGenericTorch
# from disent.nn.functional._util_generic import TypeGenericNumpy
# from disent.nn.functional._util_generic import generic_as_int32
# from disent.nn.functional._util_generic import generic_max
# from disent.nn.functional._util_generic import generic_min
# from disent.nn.functional._util_generic import generic_shape
# from disent.nn.functional._util_generic import generic_ndim
