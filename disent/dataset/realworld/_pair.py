#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

# import numpy as np
# from torch.utils.data import Dataset
# from disent.util import LengthIter
#
#
# # ========================================================================= #
# # random pairs                                                              #
# # ========================================================================= #
#
#
# class RandomPairDataset(Dataset, LengthIter):
#
#     def __init__(self, dataset: Dataset):
#         assert len(dataset) > 1, 'Dataset must be contain more than one observation.'
#         self.dataset = dataset
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         # find differing random index, nearly always this will only run once.
#         rand_idx, attempts = idx, 0
#         while rand_idx == idx:
#             rand_idx = np.random.randint(len(self.dataset))
#             attempts += 1
#             if attempts > 1000:
#                 # pretty much impossible unless your dataset is of size 1, or your prng is broken...
#                 raise RuntimeError('Unable to find random index that differs.')
#         # return elements
#         return (self.dataset[idx], idx), (self.dataset[rand_idx], rand_idx)
#
#
# # ========================================================================= #
# # Contrastive Dataset                                                       #
# # ========================================================================= #
#
#
# class PairedContrastiveDataset(Dataset, LengthIter):
#
#     def __init__(self, dataset, transform):
#         """
#         Dataset that creates a randomly transformed contrastive pair.
#
#         dataset: A dataset that extends GroundTruthData
#         transforms: transform to apply - should make use of random transforms.
#         """
#         # wrapped dataset
#         self._dataset = dataset
#         # random transforms
#         self._transform = transform
#
#     def __len__(self):
#         return len(self._dataset)
#
#     def __getitem__(self, idx):
#         x = self._dataset[idx]
#         x0 = self._transform(x)
#         x1 = self._transform(x)
#         return x0, x1
