

class ReconstructionLoss(object):

    def training_loss(self, input, target):
        """
        This loss should be applied the the training output of your
        neural network, not the final output.

        Eg. Sigmoid activations can cause numerical errors with bce loss.
            So we pass the non-activated output to this function.
        """
        raise NotImplementedError
