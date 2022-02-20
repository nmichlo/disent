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

import logging
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from numbers import Number
from pprint import pformat
from typing import Any
from typing import Dict
from typing import final
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from disent import registry
from disent.nn.modules import DisentLightningModule
from disent.schedule import Schedule
from disent.util.imports import import_obj


log = logging.getLogger(__name__)


# ========================================================================= #
# framework config                                                          #
# ========================================================================= #


class DisentConfigurable(object):

    @dataclass
    class cfg(object):
        def get_keys(self) -> list:
            return list(self.to_dict().keys())

        def to_dict(self) -> dict:
            return asdict(self)

        def __str__(self):
            return pformat(self.to_dict(), sort_dicts=False)

    def __init__(self, cfg: cfg = cfg()):
        if cfg is None:
            cfg = self.__class__.cfg()
            log.info(f'Initialised default config {cfg=} for {self.__class__.__name__}')
        super().__init__()
        assert isinstance(cfg, self.__class__.cfg), f'{cfg=} ({type(cfg)}) is not an instance of {self.__class__.cfg}'
        self.cfg = cfg


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class DisentFramework(DisentConfigurable, DisentLightningModule):

    @dataclass
    class cfg(DisentConfigurable.cfg):
        # optimizer config
        optimizer: Union[str] = 'adam'  # name in the registry, eg. `adam` OR the path to an optimizer eg. `torch.optim.Adam`
        optimizer_kwargs: Optional[Dict[str, Union[str, float, int]]] = None

    def __init__(
        self,
        cfg: cfg = None,
        # apply the batch augmentations on the GPU instead
        batch_augment: callable = None,
    ):
        # save the config values to the class
        super().__init__(cfg=cfg)
        # check the optimizer
        self.cfg.optimizer = self._check_optimizer(self.cfg.optimizer)
        self.cfg.optimizer_kwargs = self._check_optimizer_kwargs(self.cfg.optimizer_kwargs)
        # batch augmentations may not be implemented as dataset
        # transforms so we can apply these on the GPU instead
        assert callable(batch_augment) or (batch_augment is None), f'invalid batch_augment: {repr(batch_augment)}, must be callable or `None`'
        self._batch_augment = batch_augment
        # schedules
        # - maybe add support for schedules in the config?
        self._registered_schedules = set()
        self._active_schedules: Dict[str, Tuple[Any, Schedule]] = {}

    @staticmethod
    def _check_optimizer(optimizer: str):
        if not isinstance(optimizer, str):
            raise TypeError(f'invalid optimizer: {repr(optimizer)}, must be a `str`')
        # check that the optimizer has been registered
        # otherwise check that the optimizer class can be imported instead
        if optimizer not in registry.OPTIMIZERS:
            try:
                import_obj(optimizer)
            except ImportError:
                raise KeyError(f'invalid optimizer: {repr(optimizer)}, valid optimizers are: {sorted(registry.OPTIMIZERS)}, or an import path to an optimizer, eg. `torch.optim.Adam`')
        # return the updated values!
        return optimizer

    @staticmethod
    def _check_optimizer_kwargs(optimizer_kwargs: Optional[dict]):
        # check the optimizer kwargs
        assert isinstance(optimizer_kwargs, dict) or (optimizer_kwargs is None), f'invalid optimizer_kwargs type, got: {type(optimizer_kwargs)}'
        # get default kwargs OR copy
        optimizer_kwargs = dict() if (optimizer_kwargs is None) else dict(optimizer_kwargs)
        # set default values
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = 1e-3
            log.info('lr not specified in `optimizer_kwargs`, setting to default value of `1e-3`')
        # return the updated values
        return optimizer_kwargs

    @final
    def configure_optimizers(self):
        # get the optimizer
        # 1. first check if the name has been registered
        # 2. then check if the name can be imported
        if self.cfg.optimizer in registry.OPTIMIZERS:
            optimizer_cls = registry.OPTIMIZERS[self.cfg.optimizer]
        else:
            optimizer_cls = import_obj(self.cfg.optimizer)
        # check that we can call the optimizer
        if not callable(optimizer_cls):
            raise TypeError(f'unsupported optimizer type: {type(optimizer_cls)}')
        # instantiate class
        optimizer_instance = optimizer_cls(self.parameters(), **self.cfg.optimizer_kwargs)
        # check instance
        if not isinstance(optimizer_instance, torch.optim.Optimizer):
            raise TypeError(f'returned object is not an instance of torch.optim.Optimizer, got: {type(optimizer_instance)}')
        # return the optimizer
        return optimizer_instance

    @final
    def _compute_loss_step(self, batch, batch_idx, update_schedules: bool):
        try:
            # augment batch with GPU support
            if self._batch_augment is not None:
                batch = self._batch_augment(batch)
            # update the config values based on registered schedules
            if update_schedules:
                # TODO: how do we handle this in the case of the validation and test step? I think this
                #       might still give the wrong results as this is based on the trainer.global_step which
                #       may be incremented by these steps.
                self._update_config_from_schedules()
            # compute loss
            # TODO: move logging into child frameworks?
            loss = self.do_training_step(batch, batch_idx)
            # check loss values
            self._assert_valid_loss(loss)
            self.log('loss', float(loss), prog_bar=True)
            # return loss
            return loss
        except Exception as e:  # pragma: no cover
            # call in all the child processes for the best chance of clearing this...
            # remove callbacks from trainer so we aren't stuck running forever!
            # TODO: this is a hack... there must be a better way to do this... could it be a pl bug?
            #       this logic is duplicated in the run_utils
            if self.trainer and self.trainer.callbacks:
                self.trainer.callbacks.clear()
            # continue propagating errors
            raise e

    @final
    def training_step(self, batch, batch_idx):
        """This is a pytorch-lightning function that should return the computed loss"""
        return self._compute_loss_step(batch, batch_idx, update_schedules=True)

    def validation_step(self, batch, batch_idx):
        """
        TODO: how do we handle the schedule in this case?
        """
        return self._compute_loss_step(batch, batch_idx, update_schedules=False)

    def test_step(self, batch, batch_idx):
        """
        TODO: how do we handle the schedule in this case?
        """
        return self._compute_loss_step(batch, batch_idx, update_schedules=False)

    @final
    def _assert_valid_loss(self, loss):
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('The returned loss is nan or inf')
        if loss > 1e+20:
            raise ValueError(f'The returned loss: {loss:.2e} is out of bounds: > {1e+20:.0e}')

    def forward(self, batch) -> torch.Tensor:  # pragma: no cover
        """this function should return the single final output of the model, including the final activation"""
        raise NotImplementedError

    def do_training_step(self, batch, batch_idx) -> torch.Tensor:  # pragma: no cover
        """
        should return a dictionary of items to log with the key 'train_loss'
        as the variable to minimize
        """
        raise NotImplementedError

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Schedules                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @final
    def has_schedule(self, target: str):
        return target in self._registered_schedules

    @final
    def remove_schedule(self, target):
        if self.has_schedule(target):
            self._registered_schedules.remove(target)
            self._active_schedules.pop(target)
        else:
            raise KeyError(f'Cannot remove schedule for target {repr(target)} that does not exist!')

    @final
    def register_schedule(self, target: str, schedule: Schedule, logging=True) -> bool:
        """
        returns True if schedule has been activated, False if the key is
        not in the config and it has not be activated!
        """
        assert isinstance(target, str)
        assert isinstance(schedule, Schedule)
        assert type(schedule) is not Schedule
        # handle the case where a schedule for the target already exists!
        if self.has_schedule(target):
            raise KeyError(f'A schedule for target {repr(target)} has already been registered!')
        # check the target exists!
        possible_targets = {f.name for f in fields(self.cfg)}
        # register this target
        self._registered_schedules.add(target)
        # activate this schedule
        if target in possible_targets:
            initial_val = getattr(self.cfg, target)
            self._active_schedules[target] = (initial_val, schedule)
            if logging:
                log.info(f'Activating schedule for target {repr(target)} on {repr(self.__class__.__name__)}.')
            return True
        else:
            if logging:
                log.warning(f'Unactivated schedule for target {repr(target)} on {repr(self.__class__.__name__)} because the key was not found in the config.')
            return False

    @final
    def _update_config_from_schedules(self):
        if not self._active_schedules:
            return
        # log the step value
        self.log('scheduler/step', self.trainer.global_step)
        # update the values
        for target, (initial_val, scheduler) in self._active_schedules.items():
            # get the scheduled value
            new_value = scheduler.compute_value(step=self.trainer.global_step, value=initial_val)
            # update it on the config
            setattr(self.cfg, target, new_value)
            # log that things changed
            self.log(f'scheduled/{target}', new_value)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
