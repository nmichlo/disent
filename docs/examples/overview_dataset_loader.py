from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYSquaresData
from disent.dataset.sampling import GroundTruthPairOrigSampler
from disent.nn.transform import ToStandardisedTensor

# prepare the data
data = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset = DisentDataset(data, sampler=GroundTruthPairOrigSampler(), transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# iterate over single epoch
for batch in dataloader:
    (x0, x1) = batch['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
