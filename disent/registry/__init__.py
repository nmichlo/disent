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

"""
The disent registry only contains implemented versions of
classes and functions, no interfaces are included.
  - for interfaces and creating your own data types, extend
    the base classes from the various locations.

*NB* All modules and classes are lazily imported!

You can register your own modules and classes using the provided decorator:
eg. `DATASET.register(...options...)(your_function_or_class)`
"""

# from disent.registry._registry import ProvidedValue
# from disent.registry._registry import StaticImport
# from disent.registry._registry import DictProviders
# from disent.registry._registry import RegexProvidersSearch

from disent.registry._registry import StaticValue
from disent.registry._registry import LazyValue
from disent.registry._registry import LazyImport
from disent.registry._registry import Registry
from disent.registry._registry import RegistryImports
from disent.registry._registry import RegexConstructor
from disent.registry._registry import RegexRegistry


# ========================================================================= #
# DATASETS - should be synchronized with: `disent/dataset/data/__init__.py` #
# ========================================================================= #


# TODO: this is not yet used in disent.data or disent.frameworks
DATASETS: RegistryImports['torch.utils.data.Dataset'] = RegistryImports('DATASETS')
# groundtruth -- impl
DATASETS['cars3d_x128']       = LazyImport('disent.dataset.data._groundtruth__cars3d.Cars3dData')
DATASETS['cars3d']            = LazyImport('disent.dataset.data._groundtruth__cars3d.Cars3d64Data')
DATASETS['dsprites']          = LazyImport('disent.dataset.data._groundtruth__dsprites.DSpritesData')
DATASETS['mpi3d_toy']         = LazyImport('disent.dataset.data._groundtruth__mpi3d.Mpi3dData', subset='toy')
DATASETS['mpi3d_realistic']   = LazyImport('disent.dataset.data._groundtruth__mpi3d.Mpi3dData', subset='realistic')
DATASETS['mpi3d_real']        = LazyImport('disent.dataset.data._groundtruth__mpi3d.Mpi3dData', subset='real')
DATASETS['smallnorb_x96']     = LazyImport('disent.dataset.data._groundtruth__norb.SmallNorbData')
DATASETS['smallnorb']         = LazyImport('disent.dataset.data._groundtruth__norb.SmallNorb64Data')
DATASETS['shapes3d']          = LazyImport('disent.dataset.data._groundtruth__shapes3d.Shapes3dData')
# groundtruth -- impl synthetic
DATASETS['xcolumns']          = LazyImport('disent.dataset.data._groundtruth__xcolumns.XColumnsData')
DATASETS['xyobject']          = LazyImport('disent.dataset.data._groundtruth__xyobject.XYObjectData')
DATASETS['xyobject_shaded']   = LazyImport('disent.dataset.data._groundtruth__xyobject.XYObjectShadedData')
DATASETS['xysquares']         = LazyImport('disent.dataset.data._groundtruth__xysquares.XYSquaresData')
DATASETS['xysquares_minimal'] = LazyImport('disent.dataset.data._groundtruth__xysquares.XYSquaresMinimalData')
DATASETS['xysinglesquare']    = LazyImport('disent.dataset.data._groundtruth__xysquares.XYSingleSquareData')


# ========================================================================= #
# SAMPLERS - should be synchronized with:                                   #
#            `disent/dataset/sampling/__init__.py`                          #
# ========================================================================= #


# TODO: this is not yet used in disent.data or disent.frameworks
# changes here should also update
SAMPLERS: RegistryImports['disent.dataset.sampling.BaseDisentSampler'] = RegistryImports('SAMPLERS')
# [ground truth samplers]
SAMPLERS['gt_dist']         = LazyImport('disent.dataset.sampling._groundtruth__dist.GroundTruthDistSampler')
SAMPLERS['gt_pair']         = LazyImport('disent.dataset.sampling._groundtruth__pair.GroundTruthPairSampler')
SAMPLERS['gt_pair_orig']    = LazyImport('disent.dataset.sampling._groundtruth__pair_orig.GroundTruthPairOrigSampler')
SAMPLERS['gt_single']       = LazyImport('disent.dataset.sampling._groundtruth__single.GroundTruthSingleSampler')
SAMPLERS['gt_triple']       = LazyImport('disent.dataset.sampling._groundtruth__triplet.GroundTruthTripleSampler')
# [any dataset samplers]
SAMPLERS['single']          = LazyImport('disent.dataset.sampling._single.SingleSampler')
SAMPLERS['random']          = LazyImport('disent.dataset.sampling._random__any.RandomSampler')
# [episode samplers]
SAMPLERS['random_episode']  = LazyImport('disent.dataset.sampling._random__episodes.RandomEpisodeSampler')


# ========================================================================= #
# FRAMEWORKS - should be synchronized with:                                 #
#             `disent/frameworks/ae/__init__.py`                            #
#             `disent/frameworks/ae/experimental/__init__.py`               #
#             `disent/frameworks/vae/__init__.py`                           #
#             `disent/frameworks/vae/experimental/__init__.py`              #
# ========================================================================= #


# TODO: this is not yet used in disent.frameworks
FRAMEWORKS: RegistryImports['disent.frameworks.DisentFramework'] = RegistryImports('FRAMEWORKS')
# [AE]
FRAMEWORKS['tae']           = LazyImport('disent.frameworks.ae._supervised__tae.TripletAe')
FRAMEWORKS['ae']            = LazyImport('disent.frameworks.ae._unsupervised__ae.Ae')
FRAMEWORKS['ada_ae']        = LazyImport('disent.frameworks.ae._weaklysupervised__adaae.AdaAe')             # ae version of ada_vae
FRAMEWORKS['adaneg_tae']    = LazyImport('disent.frameworks.ae._supervised__adaneg_tae.AdaNegTripletAe')    # ae version of adaneg_tvae
FRAMEWORKS['adaneg_tae_d']  = LazyImport('disent.frameworks.ae._unsupervised__dotae.DataOverlapTripletAe')  # ae version of adaneg_tvae_d
# [VAE]
FRAMEWORKS['tvae']          = LazyImport('disent.frameworks.vae._supervised__tvae.TripletVae')
FRAMEWORKS['betatc_vae']    = LazyImport('disent.frameworks.vae._unsupervised__betatcvae.BetaTcVae')
FRAMEWORKS['beta_vae']      = LazyImport('disent.frameworks.vae._unsupervised__betavae.BetaVae')
FRAMEWORKS['dfc_vae']       = LazyImport('disent.frameworks.vae._unsupervised__dfcvae.DfcVae')
FRAMEWORKS['dip_vae']       = LazyImport('disent.frameworks.vae._unsupervised__dipvae.DipVae')
FRAMEWORKS['info_vae']      = LazyImport('disent.frameworks.vae._unsupervised__infovae.InfoVae')
FRAMEWORKS['vae']           = LazyImport('disent.frameworks.vae._unsupervised__vae.Vae')
FRAMEWORKS['ada_vae']       = LazyImport('disent.frameworks.vae._weaklysupervised__adavae.AdaVae')
FRAMEWORKS['adaneg_tvae']   = LazyImport('disent.frameworks.vae._supervised__adaneg_tvae.AdaNegTripletVae')
FRAMEWORKS['adaneg_tvae_d'] = LazyImport('disent.frameworks.vae._unsupervised__dotvae.DataOverlapTripletVae')


# ========================================================================= #
# RECON_LOSSES - should be synchronized with:                               #
#                `disent/frameworks/helper/reconstructions.py`              #
# ========================================================================= #


RECON_LOSSES: RegexRegistry['disent.frameworks.helper.reconstructions.ReconLossHandler'] = RegexRegistry('RECON_LOSSES')  # TODO: we need a regex version of RegistryImports
# [STANDARD LOSSES]
RECON_LOSSES['mse']         = LazyImport('disent.frameworks.helper.reconstructions.ReconLossHandlerMse')                  # from the normal distribution - real values in the range [0, 1]
RECON_LOSSES['mae']         = LazyImport('disent.frameworks.helper.reconstructions.ReconLossHandlerMae')                  # mean absolute error
# [STANDARD DISTRIBUTIONS]
RECON_LOSSES['bce']         = LazyImport('disent.frameworks.helper.reconstructions.ReconLossHandlerBce')                  # from the bernoulli distribution - binary values in the set {0, 1}
RECON_LOSSES['bernoulli']   = LazyImport('disent.frameworks.helper.reconstructions.ReconLossHandlerBernoulli')            # reduces to bce - binary values in the set {0, 1}
RECON_LOSSES['c_bernoulli'] = LazyImport('disent.frameworks.helper.reconstructions.ReconLossHandlerContinuousBernoulli')  # bernoulli with a computed offset to handle values in the range [0, 1]
RECON_LOSSES['normal']      = LazyImport('disent.frameworks.helper.reconstructions.ReconLossHandlerNormal')               # handle all real values
# [REGEX LOSSES]
RECON_LOSSES.register_regex(pattern=r'^([a-z\d]+)_([a-z\d]+_[a-z\d]+)_l(\d+\.\d+)_k(\d+\.\d+)_norm_([a-z]+)$', example='mse_xy8_abs63_l1.0_k1.0_norm_none', factory_fn='disent.frameworks.helper.reconstructions._make_aug_recon_loss_l_w_n')
RECON_LOSSES.register_regex(pattern=r'^([a-z\d]+)_([a-z\d]+_[a-z\d]+)_norm_([a-z]+)$',                         example='mse_xy8_abs63_norm_none',           factory_fn='disent.frameworks.helper.reconstructions._make_aug_recon_loss_l1_w1_n')
RECON_LOSSES.register_regex(pattern=r'^([a-z\d]+)_([a-z\d]+_[a-z\d]+)$',                                       example='mse_xy8_abs63',                     factory_fn='disent.frameworks.helper.reconstructions._make_aug_recon_loss_l1_w1_nnone')


# ========================================================================= #
# LATENT_HANDLERS - should be synchronized with:                            #
#                  `disent/frameworks/helper/latent_distributions.py`       #
# ========================================================================= #


LATENT_HANDLERS: RegistryImports['disent.frameworks.helper.latent_distributions.LatentDistsHandler'] = RegistryImports('LATENT_HANDLERS')
LATENT_HANDLERS['normal']  = LazyImport('disent.frameworks.helper.latent_distributions.LatentDistsHandlerNormal')
LATENT_HANDLERS['laplace'] = LazyImport('disent.frameworks.helper.latent_distributions.LatentDistsHandlerLaplace')


# ========================================================================= #
# OPTIMIZER                                                                 #
# ========================================================================= #


# default learning rate for each optimizer
_LR = 1e-3

OPTIMIZERS: RegistryImports['torch.optim.Optimizer'] = RegistryImports('OPTIMIZERS')
# [torch]
OPTIMIZERS['adadelta']    = LazyImport(lr=_LR, import_path='torch.optim.adadelta.Adadelta')
OPTIMIZERS['adagrad']     = LazyImport(lr=_LR, import_path='torch.optim.adagrad.Adagrad')
OPTIMIZERS['adam']        = LazyImport(lr=_LR, import_path='torch.optim.adam.Adam')
OPTIMIZERS['adamax']      = LazyImport(lr=_LR, import_path='torch.optim.adamax.Adamax')
OPTIMIZERS['adam_w']      = LazyImport(lr=_LR, import_path='torch.optim.adamw.AdamW')
OPTIMIZERS['asgd']        = LazyImport(lr=_LR, import_path='torch.optim.asgd.ASGD')
OPTIMIZERS['lbfgs']       = LazyImport(lr=_LR, import_path='torch.optim.lbfgs.LBFGS')
OPTIMIZERS['rmsprop']     = LazyImport(lr=_LR, import_path='torch.optim.rmsprop.RMSprop')
OPTIMIZERS['rprop']       = LazyImport(lr=_LR, import_path='torch.optim.rprop.Rprop')
OPTIMIZERS['sgd']         = LazyImport(lr=_LR, import_path='torch.optim.sgd.SGD')
OPTIMIZERS['sparse_adam'] = LazyImport(lr=_LR, import_path='torch.optim.sparse_adam.SparseAdam')
# [torch_optimizer]
OPTIMIZERS['acc_sgd']     = LazyImport(lr=_LR, import_path='torch_optimizer.AccSGD')
OPTIMIZERS['ada_bound']   = LazyImport(lr=_LR, import_path='torch_optimizer.AdaBound')
OPTIMIZERS['ada_mod']     = LazyImport(lr=_LR, import_path='torch_optimizer.AdaMod')
OPTIMIZERS['adam_p']      = LazyImport(lr=_LR, import_path='torch_optimizer.AdamP')
OPTIMIZERS['agg_mo']      = LazyImport(lr=_LR, import_path='torch_optimizer.AggMo')
OPTIMIZERS['diff_grad']   = LazyImport(lr=_LR, import_path='torch_optimizer.DiffGrad')
OPTIMIZERS['lamb']        = LazyImport(lr=_LR, import_path='torch_optimizer.Lamb')
# 'torch_optimizer.Lookahead' is skipped because it is wrapped
OPTIMIZERS['novograd']    = LazyImport(lr=_LR, import_path='torch_optimizer.NovoGrad')
OPTIMIZERS['pid']         = LazyImport(lr=_LR, import_path='torch_optimizer.PID')
OPTIMIZERS['qh_adam']     = LazyImport(lr=_LR, import_path='torch_optimizer.QHAdam')
OPTIMIZERS['qhm']         = LazyImport(lr=_LR, import_path='torch_optimizer.QHM')
OPTIMIZERS['radam']       = LazyImport(lr=_LR, import_path='torch_optimizer.RAdam')
OPTIMIZERS['ranger']      = LazyImport(lr=_LR, import_path='torch_optimizer.Ranger')
OPTIMIZERS['ranger_qh']   = LazyImport(lr=_LR, import_path='torch_optimizer.RangerQH')
OPTIMIZERS['ranger_va']   = LazyImport(lr=_LR, import_path='torch_optimizer.RangerVA')
OPTIMIZERS['sgd_w']       = LazyImport(lr=_LR, import_path='torch_optimizer.SGDW')
OPTIMIZERS['sgd_p']       = LazyImport(lr=_LR, import_path='torch_optimizer.SGDP')
OPTIMIZERS['shampoo']     = LazyImport(lr=_LR, import_path='torch_optimizer.Shampoo')
OPTIMIZERS['yogi']        = LazyImport(lr=_LR, import_path='torch_optimizer.Yogi')


# ========================================================================= #
# METRIC - should be synchronized with: `disent/metrics/__init__.py`        #
# ========================================================================= #


# TODO: this is not yet used in disent.util.lightning.callbacks or disent.metrics
METRICS: RegistryImports['disent.metrics.utils._Metric'] = RegistryImports('METRICS')
METRICS['dci']                 = LazyImport('disent.metrics._dci.metric_dci')
METRICS['factor_vae']          = LazyImport('disent.metrics._factor_vae.metric_factor_vae')
METRICS['mig']                 = LazyImport('disent.metrics._mig.metric_mig')
METRICS['sap']                 = LazyImport('disent.metrics._sap.metric_sap')
METRICS['unsupervised']        = LazyImport('disent.metrics._unsupervised.metric_unsupervised')
# register metrics
METRICS['flatness']            = LazyImport('disent.metrics._flatness.metric_flatness')
METRICS['factored_components'] = LazyImport('disent.metrics._factored_components.metric_factored_components')
METRICS['distances']           = LazyImport('disent.metrics._factored_components.metric_distances')
METRICS['linearity']           = LazyImport('disent.metrics._factored_components.metric_linearity')


# ========================================================================= #
# SCHEDULE - should be synchronized with: `disent/schedule/__init__.py`     #
# ========================================================================= #


# TODO: this is not yet used in disent.framework or disent.schedule
SCHEDULES: RegistryImports['disent.schedule.Schedule'] = RegistryImports('SCHEDULES')
SCHEDULES['clip']        = LazyImport('disent.schedule._schedule.ClipSchedule')
SCHEDULES['cosine_wave'] = LazyImport('disent.schedule._schedule.CosineWaveSchedule')
SCHEDULES['cyclic']      = LazyImport('disent.schedule._schedule.CyclicSchedule')
SCHEDULES['linear']      = LazyImport('disent.schedule._schedule.LinearSchedule')
SCHEDULES['noop']        = LazyImport('disent.schedule._schedule.NoopSchedule')


# ========================================================================= #
# MODEL - should be synchronized with: `disent/model/ae/__init__.py`        #
# ========================================================================= #


# TODO: this is not yet used in disent.framework or disent.model
MODELS: RegistryImports['disent.model._base.DisentLatentsModule'] = RegistryImports('MODELS')
# [DECODER]
MODELS['encoder_conv64']     = LazyImport('disent.model.ae._vae_conv64.EncoderConv64')
MODELS['encoder_conv64norm'] = LazyImport('disent.model.ae._norm_conv64.EncoderConv64Norm')
MODELS['encoder_fc']         = LazyImport('disent.model.ae._vae_fc.EncoderFC')
MODELS['encoder_linear']     = LazyImport('disent.model.ae._linear.EncoderLinear')
# [ENCODER]
MODELS['decoder_conv64']     = LazyImport('disent.model.ae._vae_conv64.DecoderConv64')
MODELS['decoder_conv64norm'] = LazyImport('disent.model.ae._norm_conv64.DecoderConv64Norm')
MODELS['decoder_fc']         = LazyImport('disent.model.ae._vae_fc.DecoderFC')
MODELS['decoder_linear']     = LazyImport('disent.model.ae._linear.DecoderLinear')


# ========================================================================= #
# HELPER registries                                                         #
# ========================================================================= #


# TODO: add norm support with regex?
KERNELS: RegexRegistry['torch.Tensor'] = RegexRegistry('KERNELS')
KERNELS.register_regex(pattern=r'^box_r(\d+)$', example='box_r31', factory_fn='disent.dataset.transform._augment._make_box_kernel')
KERNELS.register_regex(pattern=r'^gau_r(\d+)$', example='gau_r31', factory_fn='disent.dataset.transform._augment._make_gaussian_kernel')


# ========================================================================= #
# Registry of all Registries                                                #
# ========================================================================= #


# registry of registries
REGISTRIES: Registry[Registry] = Registry('REGISTRIES')
REGISTRIES['DATASETS']        = StaticValue(DATASETS)
REGISTRIES['SAMPLERS']        = StaticValue(SAMPLERS)
REGISTRIES['FRAMEWORKS']      = StaticValue(FRAMEWORKS)
REGISTRIES['RECON_LOSSES']    = StaticValue(RECON_LOSSES)
REGISTRIES['LATENT_HANDLERS'] = StaticValue(LATENT_HANDLERS)
REGISTRIES['OPTIMIZERS']      = StaticValue(OPTIMIZERS)
REGISTRIES['METRICS']         = StaticValue(METRICS)
REGISTRIES['SCHEDULES']       = StaticValue(SCHEDULES)
REGISTRIES['MODELS']          = StaticValue(MODELS)
REGISTRIES['KERNELS']         = StaticValue(KERNELS)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
