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

import json
import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional
from typing import Union

import hydra.utils
import imageio
import numpy as np
import psutil
import torch

from disent.dataset.sampling import GroundTruthRandomWalkSampler
from disent.util.visualize.vis_util import make_image_grid
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from disent.dataset import DisentIterDataset
from disent.dataset.data import GroundTruthData
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.sampling import GroundTruthDistSampler
from disent.dataset.sampling import GroundTruthPairOrigSampler
from disent.dataset.sampling import GroundTruthPairSampler
from disent.dataset.sampling import GroundTruthTripleSampler
from disent.dataset.sampling import RandomSampler
from disent.dataset.transform import ToImgTensorF32
from disent.dataset.util.stats import compute_data_mean_std
from disent.frameworks.ae import Ae
from disent.frameworks.vae import AdaVae
from disent.frameworks.vae import BetaVae
from disent.frameworks.vae import TripletVae
from disent.frameworks.vae import Vae
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.nn.weights import init_model_weights
from disent.util.lightning.callbacks import LoggerProgressCallback
from disent.util.lightning.callbacks import VaeLatentCycleLoggingCallback
from disent.util.lightning.callbacks._callback_vis_latents import get_vis_min_max
from disent.util.visualize.plot import plt_imshow
from disent.util.visualize.vis_img import torch_to_images
from experiment.util.path_utils import make_current_experiment_dir
from research.code.dataset.data import XYSingleSquareData
from research.code.frameworks.vae import AdaTripletVae
from research.code.metrics import metric_factored_components


log = logging.getLogger(__name__)


# ========================================================================= #
# Train A Single VAE                                                        #
# ========================================================================= #


def train(
    save_dir: Union[str, Path],
    data: GroundTruthData,
    sampler: BaseDisentSampler,
    framework: Union[Ae, Vae],
    train_steps: int = 5000,
    batch_size: int = 64,
    num_workers: int = psutil.cpu_count(logical=False),
    save_top_k: int = 5,
    save_every_n_steps: int = 2500,
    profile: bool = False,
    data_mean=None,
    data_std=None,
):
    vis_min, vis_max = get_vis_min_max(recon_mean=data_mean, recon_std=data_std)

    # normalise the paths
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=False, parents=True)

    # make the dataset
    dataset = DisentIterDataset(data, sampler=sampler, transform=ToImgTensorF32(size=64, mean=data_mean, std=data_std))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # make the trainer
    trainer = Trainer(
        logger=False,
        callbacks=[
            ModelCheckpoint(dirpath=save_dir, monitor='loss', verbose=True, save_top_k=save_top_k, every_n_train_steps=save_every_n_steps),
            LoggerProgressCallback(interval=5, log_level=logging.INFO),
            # RichModelSummary(max_depth=2),
            # RichProgressBar(refresh_rate_per_second=3),
            # LoggerProgressCallback(),
            # VaeMetricLoggingCallback(),
            # VaeLatentCycleLoggingCallback(),
            # VaeGtDistsLoggingCallback(),
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_steps=train_steps,
        max_epochs=train_steps,
        gpus=1 if torch.cuda.is_available() else 0,
        profiler='simple' if profile else None,
    )

    # train the framework
    trainer.fit(framework, dataloader)

    # visualise model -- generate
    stills, animation, image = VaeLatentCycleLoggingCallback.generate_visualisations(dataset, framework, recon_mean=data_mean, recon_std=data_std, mode='minmax_interval_cycle', num_stats_samples=256)

    # visualise model -- plot image, save image, save vid
    plt_imshow(image, show=True)
    imageio.imsave(save_dir.joinpath('latent_cycle.png'), image)
    with imageio.get_writer(save_dir.joinpath('latent_cycle.mp4'), fps=4) as writer:
        for frame in animation:
            writer.append_data(frame)

    # visualise data -- generate
    data_batch = dataset.dataset_sample_batch(stills.shape[0]*stills.shape[1], mode='input', seed=7777, replace=True)
    data_batch = torch_to_images(data_batch, in_min=vis_min, in_max=vis_max, always_rgb=True).numpy()
    data_image = make_image_grid(data_batch, num_cols=stills.shape[1], pad=4)

    # visualise data -- plot image, save image
    plt_imshow(data_image, show=True)
    imageio.imsave(save_dir.joinpath('data_samples.png'), data_image)

    # compute metrics
    get_repr = lambda x: framework.encode(x.to(framework.device))
    metrics = {
        # **metric_dci.compute_fast(dataset, get_repr),
        **metric_mig.compute_fast(dataset, get_repr),
        **metric_factored_components.compute_fast(dataset, get_repr),
    }

    # print and save the metrics
    pprint(metrics)
    with open(save_dir.joinpath('metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)

    # done!
    return save_dir, metrics


# ========================================================================= #
# Train Multiple VAEs -- The actual experiment!                             #
# ========================================================================= #


def _load_ada_schedules(max_steps: int):
    path = Path(__file__).parent.parent.parent.joinpath('config', 'schedule', 'adavae_up_all.yaml')
    with open(path, 'r') as fp:
        conf = DictConfig({
            'trainer': {'max_steps': max_steps},
            'schedule': OmegaConf.load(fp),
        })
    return hydra.utils.instantiate(conf.schedule.schedule_items)


def run_experiments(
    lr: float = 1e-4,
    z_size: int = 9,
    exp_dir: Optional[Union[Path, str]] = None,
    train_steps: int = 10_000,
    batch_size: int = 64,
    num_workers: int = psutil.cpu_count(logical=False),
    compute_stats: bool = False,
    profile: bool = False,
    ada_ratio: float = 1.25
):
    # PERMUTATIONS:
    datasets = [
        ('xy8', XYSingleSquareData, dict(square_size=8, grid_spacing=8, image_size=64), [0.015625], [0.12403473458920848]),
        # ('xy4', XYSingleSquareData, dict(square_size=8, grid_spacing=4, image_size=64), [0.015625], [0.12403473458920848]),
        # ('xy2', XYSingleSquareData, dict(square_size=8, grid_spacing=2, image_size=64), [0.015625], [0.12403473458920848]),
        # ('xy1', XYSingleSquareData, dict(square_size=8, grid_spacing=1, image_size=64), [0.015625], [0.12403473458920848]),
    ]
    triplet_sampler_maker_A  = lambda: GroundTruthDistSampler(num_samples=3, triplet_sample_mode='manhattan_scaled', triplet_swap_chance=0.0)
    triplet_sampler_maker_A1 = lambda: GroundTruthDistSampler(num_samples=3, triplet_sample_mode='manhattan_scaled', triplet_swap_chance=0.1)  # actually works quite well if ada_ratio is lower, eg 1.25 instead of 1.5, but might hurt recons? check?
    triplet_sampler_maker_A2 = lambda: GroundTruthDistSampler(num_samples=3, triplet_sample_mode='manhattan_scaled', triplet_swap_chance=0.2)  # actually works quite well if ada_ratio is lower, eg 1.25 instead of 1.5, but might hurt recons? check?
    triplet_sampler_maker_B  = lambda: GroundTruthTripleSampler(p_k_range=1,       n_k_range=(0, -1), n_k_sample_mode='bounded_below', n_k_is_shared=True, p_radius_range=1,       n_radius_range=(0, -1), n_radius_sample_mode='bounded_below')   # this one is really bad
    triplet_sampler_maker_C  = lambda: GroundTruthTripleSampler(p_k_range=(0, -1), n_k_range=(0, -1), n_k_sample_mode='bounded_below', n_k_is_shared=True, p_radius_range=(0, -1), n_radius_range=(0, -1), n_radius_sample_mode='bounded_below')   # pretty much the same as the manhat above, except more strict... actually less real because its bounded below. Real episodes when sampled by time will usually be further away, but not necessarily bounded below like this.
    triplet_sampler_maker_D64 = lambda: GroundTruthRandomWalkSampler(num_samples=3, p_dist_max=8, n_dist_max=64)
    triplet_sampler_maker_D32 = lambda: GroundTruthRandomWalkSampler(num_samples=3, p_dist_max=8, n_dist_max=32)
    triplet_sampler_maker_D16 = lambda: GroundTruthRandomWalkSampler(num_samples=3, p_dist_max=8, n_dist_max=16)
    triplet_sampler_maker_D8  = lambda: GroundTruthRandomWalkSampler(num_samples=3, p_dist_max=8, n_dist_max=8)

    frameworks = [
        ('betavae',   BetaVae,    lambda:        BetaVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.0001), lambda: RandomSampler(num_samples=1),                                lambda: {}),
        ('adavae',    AdaVae,     lambda:         AdaVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.0001), lambda: GroundTruthPairSampler(p_k_range=1, p_radius_range=(1, -1)), lambda: {}),
        ('adavae_os', AdaVae,     lambda:         AdaVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.0001), lambda: GroundTruthPairOrigSampler(p_k=1),                           lambda: {}),

        ('triplet_soft_A',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_A, lambda: {}),
        ('adatvae_soft_A',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_A,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),

        ('triplet_soft_A1', TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_A1, lambda: {}),
        ('triplet_soft_A2', TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_A2, lambda: {}),
        ('triplet_soft_B',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_B, lambda: {}),
        ('triplet_soft_C',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_C, lambda: {}),

        # ('triplet_sigmoid10', TripletVae, lambda:    TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_sigmoid', triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker, lambda: {}),
        # ('triplet_sigmoid1',  TripletVae, lambda:    TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_sigmoid', triplet_scale=1,  triplet_margin_max=1,  triplet_p=1),  triplet_sampler_maker, lambda: {}),
        # ('triplet10',         TripletVae, lambda:    TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet',         triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker, lambda: {}),
        # ('triplet1',          TripletVae, lambda:    TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet',         triplet_scale=1,  triplet_margin_max=1,  triplet_p=1),  triplet_sampler_maker, lambda: {}),

        ('adatvae_soft_A1',  AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_A1, lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
        ('adatvae_soft_A2',  AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_A2, lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
        ('adatvae_soft_B',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_B,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
        ('adatvae_soft_C',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_C,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),

        # ('adatvae_soft',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker, lambda: _load_ada_schedules(max_steps=int(train_steps*ADA_RATIO))),
        # ('adatvae_sig10',  AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_sigmoid', triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker, lambda: _load_ada_schedules(max_steps=int(train_steps*ADA_RATIO))),
        # ('adatvae_sig1',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_sigmoid', triplet_scale=1, triplet_margin_max=1,  triplet_p=1),  triplet_sampler_maker, lambda: _load_ada_schedules(max_steps=int(train_steps*ADA_RATIO))),
        # ('adatvae_trip10', AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet',         triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker, lambda: _load_ada_schedules(max_steps=int(train_steps*ADA_RATIO))),
        # ('adatvae_trip1',  AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet',         triplet_scale=1, triplet_margin_max=1,  triplet_p=1),  triplet_sampler_maker, lambda: _load_ada_schedules(max_steps=int(train_steps*ADA_RATIO))),

        ('triplet_soft_D64',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D64, lambda: {}),
        ('triplet_soft_D32',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D32, lambda: {}),
        ('triplet_soft_D16',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D16, lambda: {}),
        ('triplet_soft_D8',  TripletVae, lambda: TripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1,  triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D8, lambda: {}),

        ('adatvae_soft_D64',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D64,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
        ('adatvae_soft_D32',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D32,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
        ('adatvae_soft_D16',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D16,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
        ('adatvae_soft_D8',   AdaTripletVae, lambda: AdaTripletVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=lr), beta=0.001, triplet_loss='triplet_soft',    triplet_scale=1, triplet_margin_max=10, triplet_p=1),  triplet_sampler_maker_D8,  lambda: _load_ada_schedules(max_steps=int(train_steps * ada_ratio))),
    ]

    # GET NAME:
    exp_name = f'xy8_{ada_ratio}_{train_steps}'

    # NORMALIZE:
    if exp_dir is None:
        exp_dir = make_current_experiment_dir(str(Path(__file__).parent.joinpath('exp')), name=exp_name)
    exp_dir = Path(exp_dir)


    # COMPUTE DATASET STATS:
    if compute_stats:
        for i, (data_name, data_cls, data_kwargs, data_mean, data_std) in enumerate(datasets):
            data = data_cls(transform=ToImgTensorF32(), **data_kwargs)
            mean, std = compute_data_mean_std(data, progress=True, num_workers=num_workers, batch_size=batch_size)
            print(f'{data.__class__.__name__} - {data_name}:{len(data)} - {data_kwargs}:\n    mean: {mean.tolist()}\n    std: {std.tolist()}')

    # TRAIN DIFFERENT FRAMEWORKS ON DATASETS
    for i, (data_name, data_cls, data_kwargs, data_mean, data_std) in enumerate(datasets):
        # make the dataset
        data: GroundTruthData = data_cls(**data_kwargs)

        # train the framework
        for j, (framework_name, framework_cls, framework_cfg_maker, sampler_maker, schedules_maker) in enumerate(frameworks):
            start_time = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')

            # make the data & framework
            framework: Union[Ae, Vae] = framework_cls(
                model=AutoEncoder(
                    encoder=EncoderConv64(x_shape=data.x_shape, z_size=z_size, z_multiplier=2 if issubclass(framework_cls, Vae) else 1),
                    decoder=DecoderConv64(x_shape=data.x_shape, z_size=z_size),
                ),
                cfg=framework_cfg_maker()
            )
            sampler = sampler_maker()

            print('=' * 100)
            print(f'[{i}x{j}] Dataset: name={data_name}, size={len(data)}, kwargs={data_kwargs}, mean={data_mean}, std={data_std}')
            print(f'[{i}x{j}] Framework: name={framework_name} ({framework.__class__.__name__}), sampler={sampler.__class__.__name__}, cfg={framework.cfg.to_dict()}')
            print('=' * 100)

            # register the schedules
            for k, schedule in schedules_maker().items():
                framework.register_schedule(k, schedule)
            # initialize weights
            init_model_weights(framework, mode='xavier_normal', log_level=logging.DEBUG)

            # train the framework
            run_name = f'{i}x{j}_{data_name}_{framework_name}'
            save_dir = exp_dir.joinpath(run_name)
            save_dir, metrics = train(
                save_dir=save_dir,
                data=data,
                sampler=sampler,
                framework=framework,
                train_steps=train_steps,
                batch_size=batch_size,
                num_workers=num_workers,
                profile=profile,
                save_every_n_steps=train_steps,
            )

            # generate data for rl
            with torch.no_grad():
                dataset: GroundTruthData = data_cls(**data_kwargs, transform=ToImgTensorF32(size=64, mean=data_mean, std=data_std))
                dat = {
                    # experiment info
                    'exp_name': save_dir.parent.name,
                    'run_name': save_dir.name,
                    'start_time': start_time,
                    # dataset data
                    'factor_names': data.factor_names,
                    'factor_sizes': data.factor_sizes,
                    # sizes
                    'num_factors': data.num_factors,
                    'num_obs':     len(data),
                    'num_latents': z_size,
                    # image data
                    'obs_indices': np.array([i                  for i in range(len(data))]),
                    'obs_factors': np.array([data.idx_to_pos(i) for i in range(len(data))]),
                    'obs':         np.array([data[i]            for i in range(len(data))]),
                    # representations
                    'obs_encodings': np.array([framework.encode(dataset[i][None, ...].to(framework.device))[0].cpu().numpy() for i in range(len(data))]),
                    # results
                    'metrics': metrics,
                    # descriptions
                    '_desc_': {
                        # experiment info
                        'exp_name':      f'The name of the group of experiments',
                        'run_name':      f'The name of this individual run that is part of the experiment',
                        'start_time':    f'The starting time of this individual run',
                        # dataset data
                        'factor_names':  f'The names of the ground truth factors in the dataset, eg. ["x", "y"]',
                        'factor_sizes':  f'The sizes of the ground truth factors in the dataset, eg. [8, 8]',
                        # sizes
                        'num_factors':   f'The number of different ground_truth factors in the dataset, eg. 2',
                        'num_obs':       f'The number of elements in the dataset, equal to the product of all the factor sizes, eg. 8x8 = 64',
                        'num_latents':   f'The number of latent units of the model or rather the number of encoder outputs, eg. 9',
                        # image data
                        'obs_indices':   f'The index of each observation in the dataset, eg. [0, 1, ..., 62, 63]. '
                                         f'The shape is (num_obs,)',
                        'obs_factors':   f'The ground truth factor of each observation in the dataset, eg. [[0, 0], [0, 1], ..., [7, 6], [7, 7]]. '
                                         f'The shape is (num_obs, num_factors)',
                        'obs':           f'The raw observations from the dataset, eg. [<img0>, ..., <img63>]. '
                                         f'The shape is (num_obs, H, W, C)',
                        'obs_encodings': f'The low dimensional encodings of each observations, eg. [<enc1>, ..., <enc63>]. '
                                         f'The shape is (num_obs, num_latents)',
                        # results
                        'metrics':       f'Dict[str, float] of various scores from different disentanglement metrics computed over the model and the data.',
                    }
                }

            # save the data for rl
            # | rl_dat_path = save_dir.joinpath('rl_data.json')
            # | with open(rl_dat_path, 'w') as fp:
            # |     json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dat.items()}, fp)
            # | print(f'[SAVED DATA]: {rl_dat_path}')
            rl_npz_path = save_dir.joinpath('rl_data.npz')
            np.savez_compressed(rl_npz_path, data=dat)
            print(f'[SAVED DATA]: {rl_npz_path}')


# ========================================================================= #
# RUN!                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # make sure we can see the output!
    logging.basicConfig(level=logging.INFO)
    # run everything!
    # run_experiments(train_steps=10000, ada_ratio=1.5)  # ada not always strong enough? or is that a metric error
    run_experiments(train_steps=10000, ada_ratio=1.25)   # seems best?
    # run_experiments(train_steps=5000, ada_ratio=1.5)
    # run_experiments(train_steps=5000, ada_ratio=1.25)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
