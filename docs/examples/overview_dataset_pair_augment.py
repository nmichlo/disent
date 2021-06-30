from disent.dataset import DisentDataset
from disent.dataset.data import XYSquaresData
from disent.dataset.sampling import GroundTruthPairSampler
from disent.nn.transform import ToStandardisedTensor, FftBoxBlur


# prepare the data
data = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset = DisentDataset(data, sampler=GroundTruthPairSampler(), transform=ToStandardisedTensor(), augment=FftBoxBlur(radius=1, p=1.0))

# iterate over single epoch
for obs in dataset:
    # if augment is not specified, then the augmented 'x' key does not exist!
    (x0, x1), (x0_targ, x1_targ) = obs['x'], obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
