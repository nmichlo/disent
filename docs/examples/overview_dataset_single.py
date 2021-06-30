from torch.utils.data import Dataset
from disent.data.groundtruth import GroundTruthData, XYSquaresData
from disent.dataset import DisentGroundTruthSamplingDataset
from disent.dataset.groundtruth import GroundTruthSingleSampler


data: GroundTruthData = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset: Dataset = DisentGroundTruthSamplingDataset(data, sampler=GroundTruthSingleSampler(), transform=None, augment=None)

for obs in dataset:
    # transform is applied to data to get x_targ, then augment to get x
    # if augment is None then 'x' doesn't exist in the obs
    (x0,) = obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
