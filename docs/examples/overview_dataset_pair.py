from disent.dataset import DisentDataset
from disent.dataset.data import XYSquaresData
from disent.dataset.sampling import GroundTruthPairOrigSampler
from disent.nn.transform import ToStandardisedTensor


# prepare the data
data = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset = DisentDataset(data, sampler=GroundTruthPairOrigSampler(), transform=ToStandardisedTensor())

# iterate over single epoch
for obs in dataset:
    # singles are contained in tuples of size 1 for compatibility with pairs with size 2
    (x0, x1) = obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
