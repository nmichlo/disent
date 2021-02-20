from pprint import pprint

from eunomia import eunomia_load
from eunomia.config.nodes import SubNode
from eunomia.registry import RegistryGroup


# ========================================================================= #
# eunomia helper                                                            #
# ========================================================================= #


REGISTRY = RegistryGroup()


# ========================================================================= #
# register: disent                                                          #
# ========================================================================= #

# data type -- what kind of data is being used, this affects what type of data wrapper is needed for the framework
data_type = REGISTRY.get_group_from_path('auto/data_type', make_missing=True)
type_ground_truth = data_type.new_option('ground_truth', data=dict(source='ground_truth'))
type_episodes     = data_type.new_option('episodes',     data=dict(source='episodes'))

import disent.data.groundtruth
import disent.data.episodes

REGISTRY.register_target(disent.data.groundtruth.Cars3dData,      name='cars3d',    path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.DSpritesData,    name='dsprites',  path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.Mpi3dData,       name='mpi3d',     path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.SmallNorbData,   name='smallnorb', path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.Shapes3dData,    name='shapes3d',  path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.XYObjectData,    name='xyobject',  path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.XYSquaresData,   name='xysquares', path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(disent.data.groundtruth.XYBlocksData,    name='xyblocks',  path='disent/data', defaults=[type_ground_truth])
REGISTRY.register_target(
    disent.data.episodes.OptionEpisodesDownloadZippedPickledData, name='monte_rollouts', path='disent/data', defaults=[type_episodes],
    params=dict(
        required_file='${dataset.data_dir}/episodes/monte.pkl',
        download_url='https://raw.githubusercontent.com/nmichlo/uploads/main/monte_key.tar.xz'
    )
)


# ========================================================================= #
# register: disent.dataset                                                  #
# ========================================================================= #


import disent.dataset.groundtruth
import disent.dataset.episodes

REGISTRY.register_target(disent.dataset.groundtruth.GroundTruthDataset,              name='ground_truth_single',     path='auto/data_wrapper', params=dict())
REGISTRY.register_target(disent.dataset.groundtruth.GroundTruthDatasetPairs,         name='ground_truth_pairs',      path='auto/data_wrapper', params=dict())
REGISTRY.register_target(disent.dataset.groundtruth.GroundTruthDatasetOrigWeakPairs, name='ground_truth_weak_pairs', path='auto/data_wrapper', params=dict())
REGISTRY.register_target(disent.dataset.groundtruth.GroundTruthDatasetTriples,       name='ground_truth_triples',    path='auto/data_wrapper', params=dict())
REGISTRY.register_target(disent.dataset.episodes.RandomEpisodeDataset,               name='episodes_single',         path='auto/data_wrapper', params=dict())  # num_samples=1, sample_radius=32))
REGISTRY.register_target(disent.dataset.episodes.RandomEpisodeDataset,               name='episodes_pairs',          path='auto/data_wrapper', params=dict())  # num_samples=2, sample_radius=32))
REGISTRY.register_target(disent.dataset.episodes.RandomEpisodeDataset,               name='episodes_triples',        path='auto/data_wrapper', params=dict())  # num_samples=3, sample_radius=32))


# ========================================================================= #
# register: disent.frameworks                                               #
# ========================================================================= #


from disent.frameworks.ae  import unsupervised     as ae_unsupervised
from disent.frameworks.vae import supervised       as vae_supervised
from disent.frameworks.vae import weaklysupervised as vae_weaklysupervised
from disent.frameworks.vae import unsupervised     as vae_unsupervised

# data wrap mode -- how many inputs needs to be sampled to form a single observation from the dataset
data_wrap_mode = REGISTRY.get_group_from_path('auto/data_wrap', make_missing=True)
wrap_triples = data_wrap_mode.new_option('triples', data=dict(count=3, mode='triples'), defaults=[SubNode('/auto/data_wrapper/${/auto/data_type}_${/auto/data_wrap}')])
wrap_pairs   = data_wrap_mode.new_option('pairs',   data=dict(count=2, mode='pairs'),   defaults=[SubNode('/auto/data_wrapper/${/auto/data_type}_${/auto/data_wrap}')])
wrap_single  = data_wrap_mode.new_option('single',  data=dict(count=1, mode='single'),  defaults=[SubNode('/auto/data_wrapper/${/auto/data_type}_${/auto/data_wrap}')])

REGISTRY.register_target(ae_unsupervised.AE.cfg,                        name='ae',          path='disent/framework', nest_path='cfg', defaults=[wrap_single])
REGISTRY.register_target(vae_unsupervised.Vae.cfg,                      name='vae',         path='disent/framework', nest_path='cfg', defaults=[wrap_single])
REGISTRY.register_target(vae_unsupervised.BetaVae.cfg,                  name='betavae',     path='disent/framework', nest_path='cfg', defaults=[wrap_single])
REGISTRY.register_target(vae_unsupervised.BetaVaeH.cfg,                 name='betavae_h',   path='disent/framework', nest_path='cfg', defaults=[wrap_single])
REGISTRY.register_target(vae_unsupervised.DfcVae.cfg,                   name='dfcvae',      path='disent/framework', nest_path='cfg', defaults=[wrap_single])
REGISTRY.register_target(vae_weaklysupervised.AdaVae.cfg,               name='adavae',      path='disent/framework', nest_path='cfg', defaults=[wrap_pairs])
REGISTRY.register_target(vae_weaklysupervised.SwappedTargetAdaVae.cfg,  name='st_adavae',   path='disent/framework', nest_path='cfg', defaults=[wrap_pairs])
REGISTRY.register_target(vae_weaklysupervised.SwappedTargetBetaVae.cfg, name='st_betavae',  path='disent/framework', nest_path='cfg', defaults=[wrap_pairs])
REGISTRY.register_target(vae_weaklysupervised.AugPosTripletVae.cfg,     name='augpos_tvae', path='disent/framework', nest_path='cfg', defaults=[wrap_pairs])
REGISTRY.register_target(vae_supervised.TripletVae.cfg,                 name='tvae',        path='disent/framework', nest_path='cfg', defaults=[wrap_triples])
REGISTRY.register_target(vae_supervised.BoundedAdaVae.cfg,              name='b_adavae',    path='disent/framework', nest_path='cfg', defaults=[wrap_triples])
REGISTRY.register_target(vae_supervised.GuidedAdaVae.cfg,               name='g_adavae',    path='disent/framework', nest_path='cfg', defaults=[wrap_triples])
REGISTRY.register_target(vae_supervised.TripletBoundedAdaVae.cfg,       name='tb_adavae',   path='disent/framework', nest_path='cfg', defaults=[wrap_triples])
REGISTRY.register_target(vae_supervised.TripletGuidedAdaVae.cfg,        name='tg_adavae',   path='disent/framework', nest_path='cfg', defaults=[wrap_triples])
REGISTRY.register_target(vae_supervised.AdaTripletVae.cfg,              name='ada_tvae',    path='disent/framework', nest_path='cfg', defaults=[wrap_triples])


# ========================================================================= #
# register: disent.metrics                                                  #
# ========================================================================= #


import disent.metrics

REGISTRY.register_target(disent.metrics.metric_dci,          path='disent/metric')
REGISTRY.register_target(disent.metrics.metric_factor_vae,   path='disent/metric')
REGISTRY.register_target(disent.metrics.metric_mig,          path='disent/metric')
REGISTRY.register_target(disent.metrics.metric_sap,          path='disent/metric')
REGISTRY.register_target(disent.metrics.metric_unsupervised, path='disent/metric')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

REGISTRY.debug_tree_print(full_option_path=True, show_defaults=True)

REGISTRY.new_option('default', defaults=[
    ('/disent/data', 'dsprites'),
    ('/disent/framework', 'adavae'),
])

from ruamel import yaml

print(yaml.round_trip_dump(eunomia_load(REGISTRY)))
