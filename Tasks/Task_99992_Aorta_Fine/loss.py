import warnings

import torch.nn as nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class CustomLoss(nn.Module):
    def __init__(self, loss, weights):
        super(CustomLoss, self).__init__()
        print(f"==> Using custom loss from {__file__}")

        self.loss = DeepSupervisionWrapper(loss, weights)
        self.first_iter = True

    def forward(self, net_output, target):
        if self.first_iter:
            warnings.warn(f"==> Model output shape is {net_output[0].shape}")
            self.first_iter = False
        return self.loss(net_output, target)
