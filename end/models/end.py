import torch
import torch.nn as nn
from wide_resnet import ToricWideResNet
from symmetries_pool import TranslationPool

class End(nn.Module):
    """ Equivariant Wide-RestNet implementation for END decoder"""

    def __init__(self):
        super().__init__()
        
        self.toric_wrn = ToricWideResNet(init_channels=128)
        self.trans_pool = TranslationPool()

    def forward(self, x):
        x_pred = self.toric_wrn(x)
        classes_pred = self.trans_pool(x,x_pred)
        return classes_pred

    