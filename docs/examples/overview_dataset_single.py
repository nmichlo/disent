from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData

# prepare the data
# - DisentDataset is a generic wrapper around torch Datasets that prepares
#   the data for the various frameworks according to some sampling strategy
#   by default this sampling strategy just returns the data at the given idx.
data = XYObjectData(grid_size=4, min_square_size=1, max_square_size=2, square_size_spacing=1, palette="rgb_1")
dataset = DisentDataset(data, transform=None, augment=None)

# iterate over single epoch
for obs in dataset:
    # transform(data[i]) gives 'x_targ', then augment(x_targ) gives 'x'
    (x0,) = obs["x_targ"]
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
