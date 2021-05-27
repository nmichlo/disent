from torch.utils.data import Dataset
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset

data: GroundTruthData = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset: Dataset = GroundTruthDataset(data, transform=None, augment=None)

for obs in dataset:
    # transform is applied to data to get x_targ, then augment to get x
    # if augment is None then 'x' doesn't exist in the obs
    (x0,) = obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)