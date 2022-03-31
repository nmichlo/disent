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
    # groundtruth -- impl synthetic
    R.DATASETS.setm['xyblocks']          = R.LazyImport('research.code.dataset.data._groundtruth__xyblocks.XYBlocksData')

    # [VAE - EXPERIMENTAL]
    R.FRAMEWORKS.setm['x__adaave_tvae'] = R.LazyImport('research.code.frameworks.vae._supervised__adaave_tvae.AdaAveTripletVae')
    R.FRAMEWORKS.setm['x__ada_tvae']    = R.LazyImport('research.code.frameworks.vae._supervised__adatvae.AdaTripletVae')
    R.FRAMEWORKS.setm['x__bada_vae']    = R.LazyImport('research.code.frameworks.vae._supervised__badavae.BoundedAdaVae')
    R.FRAMEWORKS.setm['x__gada_vae']    = R.LazyImport('research.code.frameworks.vae._supervised__gadavae.GuidedAdaVae')
    R.FRAMEWORKS.setm['x__tbada_vae']   = R.LazyImport('research.code.frameworks.vae._supervised__tbadavae.TripletBoundedAdaVae')
    R.FRAMEWORKS.setm['x__tgada_vae']   = R.LazyImport('research.code.frameworks.vae._supervised__tgadavae.TripletGuidedAdaVae')
    R.FRAMEWORKS.setm['x__dor_vae']     = R.LazyImport('research.code.frameworks.vae._unsupervised__dorvae.DataOverlapRankVae')
    R.FRAMEWORKS.setm['x__augpos_tvae'] = R.LazyImport('research.code.frameworks.vae._weaklysupervised__augpostriplet.AugPosTripletVae')
    R.FRAMEWORKS.setm['x__st_ada_vae']  = R.LazyImport('research.code.frameworks.vae._weaklysupervised__st_adavae.SwappedTargetAdaVae')
    R.FRAMEWORKS.setm['x__st_beta_vae'] = R.LazyImport('research.code.frameworks.vae._weaklysupervised__st_betavae.SwappedTargetBetaVae')
