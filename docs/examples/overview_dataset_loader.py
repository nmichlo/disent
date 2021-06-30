from torch.utils.data import DataLoader, Dataset
from disent.data.groundtruth import GroundTruthData, XYSquaresData
from disent.dataset.groundtruth import GroundTruthPairSampler
from disent.dataset import DisentGroundTruthSamplingDataset
from disent.nn.transform import ToStandardisedTensor


data: GroundTruthData = XYSquaresData(square_size=1, image_size=2, num_squares=2)
dataset: Dataset = DisentGroundTruthSamplingDataset(data, GroundTruthPairSampler(), transform=ToStandardisedTensor(), augment=None)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    (x0, x1) = batch['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
