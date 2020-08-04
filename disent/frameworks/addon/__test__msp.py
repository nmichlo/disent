import torch
from disent.frameworks.addon.msp import MatrixSubspaceProjection


def test_msp_orthonormal():
    batch_size, z_size, y_size = 128, 6, 6
    msp = MatrixSubspaceProjection(y_size=y_size, z_size=z_size, init_mode='ortho')

    # new random 'activation' from encoder
    z = torch.randn((batch_size, z_size))

    # reset all labels in the label space
    labels = torch.as_tensor([0, 0, 0, 0, 0, 0])
    apply_mask = torch.as_tensor([True, True, True, True, True, True])
    z_mutated = msp.mutated_z(z, labels, apply_mask)

    # assert latent space is also zero after mutating due to 'ortho' init
    # typically this is achieved with loss on the M matrix
    assert torch.allclose(z_mutated, torch.zeros_like(z_mutated), atol=1e-05)
