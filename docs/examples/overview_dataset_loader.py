from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import GroundTruthPairOrigSampler
from disent.dataset.transform import ToImgTensorF32

# prepare the data
data = XYObjectData(grid_size=4, min_square_size=1, max_square_size=2, square_size_spacing=1, palette="rgb_1")
dataset = DisentDataset(data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# iterate over single epoch
for batch in dataloader:
    (x0, x1) = batch["x_targ"]
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
