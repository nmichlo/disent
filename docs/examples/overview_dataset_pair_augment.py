from torch.utils.data import Dataset
from disent.dataset.data.groundtruth import GroundTruthData, XYSquaresData
from disent.dataset import DisentGroundTruthSamplingDataset
from disent.dataset.samplers.groundtruth import GroundTruthPairSampler
from disent.nn.transform import FftBoxBlur, ToStandardisedTensor


data: GroundTruthData = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset: Dataset = DisentGroundTruthSamplingDataset(data, sampler=GroundTruthPairSampler(), transform=ToStandardisedTensor(), augment=FftBoxBlur(radius=1, p=1.0))

for obs in dataset:
    # if augment is not None so the augmented 'x' exists in the observation
    (x0, x1), (x0_targ, x1_targ) = obs['x'], obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
