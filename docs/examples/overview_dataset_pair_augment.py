from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import GroundTruthPairSampler
from disent.dataset.transform import FftBoxBlur
from disent.dataset.transform import ToImgTensorF32

# prepare the data
data = XYObjectData(grid_size=4, min_square_size=1, max_square_size=2, square_size_spacing=1, palette="rgb_1")
dataset = DisentDataset(
    data, sampler=GroundTruthPairSampler(), transform=ToImgTensorF32(), augment=FftBoxBlur(radius=1, p=1.0)
)

# iterate over single epoch
for obs in dataset:
    # if augment is not specified, then the augmented 'x' key does not exist!
    (x0, x1), (x0_targ, x1_targ) = obs["x"], obs["x_targ"]
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
