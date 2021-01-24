from torch.utils.data import Dataset
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset

data: GroundTruthData = XYSquaresData(square_size=1, grid_size=2, num_squares=2)
dataset: Dataset = GroundTruthDataset(data, transform=None, augment=None)

for obs in dataset:
    # transform is applied to data to get x, then augment to get x_targ
    # if augment is None then x is x_targ
    (x0,), (x0_targ,) = obs['x'], obs['x_targ']
    print(x0 is x0_targ, x0.dtype, x0.min(), x0.max(), x0.shape)