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

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch

from disent.util.iters import aggregate_dict
from disent.util.iters import collect_dicts
from disent.util.iters import map_all


# ========================================================================= #
# AVE LOSS HELPER                                                           #
# ========================================================================= #


def detach_all(tensors: Sequence[torch.tensor], if_: bool = True):
    if if_:
        return tuple(tensor.detach() for tensor in tensors)
    return tensors


# ========================================================================= #
# AVE LOSS HELPER                                                           #
# ========================================================================= #


def compute_ave_loss(loss_fn, *arg_list, **common_kwargs) -> torch.Tensor:
    # compute all losses
    losses = map_all(loss_fn, *arg_list, collect_returned=False, common_kwargs=common_kwargs)
    # compute mean loss
    loss = torch.stack(losses).mean(dim=0)
    # return!
    return loss


def compute_ave_loss_and_logs(loss_and_logs_fn, *arg_list, **common_kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # compute all losses
    losses, logs = map_all(loss_and_logs_fn, *arg_list, collect_returned=True, common_kwargs=common_kwargs)
    # compute mean loss
    loss = torch.stack(losses).mean(dim=0)
    # compute mean logs
    logs = aggregate_dict(collect_dicts(logs))
    # return!
    return loss, logs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

