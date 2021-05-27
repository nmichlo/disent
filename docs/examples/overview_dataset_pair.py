from torch.utils.data import Dataset
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDatasetPairs
from disent.transform import ToStandardisedTensor

data: GroundTruthData = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset: Dataset = GroundTruthDatasetPairs(data, transform=ToStandardisedTensor(), augment=None)

for obs in dataset:
    # singles are contained in tuples of size 1 for compatibility with pairs with size 2
    (x0, x1) = obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)