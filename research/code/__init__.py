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

import disent.registry as R


def register_to_disent():
    # register metrics
    R.METRICS.setm['flatness']            = R.LazyImport('research.code.metrics._flatness.metric_flatness')
    R.METRICS.setm['flatness_components'] = R.LazyImport('research.code.metrics._flatness_components.metric_flatness_components')
    R.METRICS.setm['distances']           = R.LazyImport('research.code.metrics._flatness_components.metric_distances')
    R.METRICS.setm['linearity']           = R.LazyImport('research.code.metrics._flatness_components.metric_linearity')

    # groundtruth -- impl synthetic
    R.DATASETS.setm['xyblocks']          = R.LazyImport('research.code.dataset.data._groundtruth__xyblocks')
    R.DATASETS.setm['xysquares']         = R.LazyImport('research.code.dataset.data._groundtruth__xysquares')
    R.DATASETS.setm['xysquares_minimal'] = R.LazyImport('research.code.dataset.data._groundtruth__xysquares')
    R.DATASETS.setm['xcolumns']          = R.LazyImport('research.code.dataset.data._groundtruth__xcolumns')

    # [AE - EXPERIMENTAL]
    R.FRAMEWORKS.setm['x__adaneg_tae']  = R.LazyImport('research.code.frameworks.ae._supervised__adaneg_tae.AdaNegTripletAe')
    R.FRAMEWORKS.setm['x__dot_ae']      = R.LazyImport('research.code.frameworks.ae._unsupervised__dotae.DataOverlapTripletAe')
    R.FRAMEWORKS.setm['x__ada_ae']      = R.LazyImport('research.code.frameworks.ae._weaklysupervised__adaae.AdaAe')

    # [VAE - EXPERIMENTAL]
    R.FRAMEWORKS.setm['x__adaave_tvae'] = R.LazyImport('research.code.frameworks.vae._supervised__adaave_tvae.AdaAveTripletVae')
    R.FRAMEWORKS.setm['x__adaneg_tvae'] = R.LazyImport('research.code.frameworks.vae._supervised__adaneg_tvae.AdaNegTripletVae')
    R.FRAMEWORKS.setm['x__ada_tvae']    = R.LazyImport('research.code.frameworks.vae._supervised__adatvae.AdaTripletVae')
    R.FRAMEWORKS.setm['x__bada_vae']    = R.LazyImport('research.code.frameworks.vae._supervised__badavae.BoundedAdaVae')
    R.FRAMEWORKS.setm['x__gada_vae']    = R.LazyImport('research.code.frameworks.vae._supervised__gadavae.GuidedAdaVae')
    R.FRAMEWORKS.setm['x__softada_tvae']= R.LazyImport('research.code.frameworks.vae._supervised__softadatvae.SoftAdaTripletVae')
    R.FRAMEWORKS.setm['x__tbada_vae']   = R.LazyImport('research.code.frameworks.vae._supervised__tbadavae.TripletBoundedAdaVae')
    R.FRAMEWORKS.setm['x__tgada_vae']   = R.LazyImport('research.code.frameworks.vae._supervised__tgadavae.TripletGuidedAdaVae')
    R.FRAMEWORKS.setm['x__dor_vae']     = R.LazyImport('research.code.frameworks.vae._unsupervised__dorvae.DataOverlapRankVae')
    R.FRAMEWORKS.setm['x__dot_vae']     = R.LazyImport('research.code.frameworks.vae._unsupervised__dotvae.DataOverlapTripletVae')
    R.FRAMEWORKS.setm['x__augpos_tvae'] = R.LazyImport('research.code.frameworks.vae._weaklysupervised__augpostriplet.AugPosTripletVae')
    R.FRAMEWORKS.setm['x__st_ada_vae']  = R.LazyImport('research.code.frameworks.vae._weaklysupervised__st_adavae.SwappedTargetAdaVae')
    R.FRAMEWORKS.setm['x__st_beta_vae'] = R.LazyImport('research.code.frameworks.vae._weaklysupervised__st_betavae.SwappedTargetBetaVae')

    # register the kernels for the loss functions!
    R.KERNELS.setm.register_regex(pattern=r'^(xy8)_r(47)$', example='xy8_r47', factory_fn='research.code.dataset.transform._augment._make_xy8_r47')
    R.KERNELS.setm.register_regex(pattern=r'^(xy1)_r(47)$', example='xy1_r47', factory_fn='research.code.dataset.transform._augment._make_xy1_r47')
