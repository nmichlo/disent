# import torch
# import torch.utils.data
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torchvision import transforms
# import torch_optimizer as optim_extra
# from dataset import Shapes3dDataset, make_datasets
# from loss import AdaGVaeLoss, BetaVaeLoss, VaeLoss
# from model import VAE
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # ========================================================================= #
# # Train & Test Steps                                                        #
# # ========================================================================= #
#
#
# def train(vae, loss_function, optimizer, train_loader, epoch, print_steps=512):
#     vae.train()
#     train_loss = 0
#     for batch_idx, (x, _) in enumerate(train_loader):
#         if torch.cuda.is_available():
#             x = x.cuda()
#         optimizer.zero_grad()
#
#         x_recon, z_mean, z_logvar = vae(x)
#         loss = loss_function(x, x_recon, z_mean, z_logvar)
#
#         loss.backward()
#         train_loss += loss.item()
#
#         print(loss.item())
#
#         optimizer.step()
#
#         if batch_idx % print_steps == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(x), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(x)))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
#
#
# def test(vae, loss_function, test_loader):
#     vae.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for x, _ in test_loader:
#             if torch.cuda.is_available():
#                 x = x.cuda()
#             x_recon, mu, log_var = vae(x)
#             # sum up batch loss
#             test_loss += loss_function(x, x_recon, mu, log_var).item()
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
#
#
# def train_loop(vae, loss, optimizer, train_loader, test_loader=None, epochs=3):
#     for epoch in range(1, epochs+1):
#         print(f'[Epoch]: {epoch}')
#         train(vae, loss, optimizer, train_loader, epoch)
#         if test_loader:
#             test(vae, loss, test_loader)
#     return vae
#
# # ========================================================================= #
# # Utility                                                                   #
# # ========================================================================= #
#
#
#
#
# def display_generated_grid(vae, seed=777):
#     seed_all(seed)
#     with torch.no_grad():
#         # GENERATE SAMPLES
#         z = torch.randn(64, 2).cuda()
#         samples = vae.decoder(z).cuda()
#         images = samples.view(64, 1, 28, 28).reshape(-1, 28, 28).cpu().detach().numpy()
#
#         # DISPLAY IMAGES
#         fig, axs = plt.subplots(8, 8)
#         axs = np.array(axs).reshape(-1)
#         for ax, img in zip(axs, images):
#             ax.imshow(img)
#         plt.show()
#
# def seed_all(seed=777):
#     """
#     https://pytorch.org/docs/stable/notes/randomness.html
#     """
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     print(f'[SEEDED]: {seed}')
#
# def display_grid(vae, seed=777):
#     seed_all(seed)
#     display_generated_grid(vae)
#
# # Make System
# def make_system(loss, optimizer, seed=777, batch_size=64, dataset='mnist'):
#     if seed is not None:
#         seed_all(seed)
#
#     # DATASETS
#     train_dataset, test_dataset = make_datasets(dataset)
#
#     # DATA LOADERS
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
#     # MAKE MODEL
#     vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
#     if torch.cuda.is_available():
#         vae.cuda()
#     else:
#         print('[WARNING]: CUDA not available!')
#
#     # MAKE OPTIMISER
#     if optimizer == 'radam':
#         optimizer = optim_extra.RAdam(vae.parameters())
#     else:
#         raise KeyError(f'Unsupported Optimizer: {optimizer}')
#
#     # MAKE LOSS
#     if loss == 'vae':
#         loss = VaeLoss()
#     elif loss == 'beta-vae':
#         loss = BetaVaeLoss()
#     elif loss == 'ada-gvae':
#         def sampler():
#             samples = iter(train_loader).next()[0]
#             if torch.cuda.is_available():
#                 samples = samples.cuda()
#             return samples
#         loss = AdaGVaeLoss(vae=vae, sampler=sampler)
#     else:
#         raise KeyError(f'Unsupported Loss: {loss}')
#
#     return vae, loss, optimizer, train_loader, test_loader
#
#
# # ========================================================================= #
# # Main                                                                       #
# # ========================================================================= #
#
#
# if __name__ == "__main__":
#     # TRAIN
#     vae = train_loop(*make_system('vae', 'radam', seed=777, batch_size=64, dataset='mnist'), epochs=3)
#     display_generated_grid(vae)
#
#
# # ========================================================================= #
# # END                                                                       #
# # ========================================================================= #



