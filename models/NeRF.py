import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class NeRF(nn.Module):
    
    def __init__(self, input_dim=60, layer_dim=256, activation_fn='leaky_relu'):
        super(NeRF, self).__init__()

        self.input_dim = input_dim
        self.layer_dim = layer_dim

        if activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation_fn == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise NotImplementedError

        self.layer_0 = nn.Linear(self.input_dim, self.layer_dim)
        self.layer_1 = nn.Linear(self.layer_dim, self.layer_dim)
        self.layer_2 = nn.Linear(self.layer_dim, self.layer_dim)
        self.layer_3 = nn.Linear(self.layer_dim, self.layer_dim)
        self.layer_4 = nn.Linear(self.layer_dim, self.layer_dim)
        self.layer_5 = nn.Linear(self.input_dim + self.layer_dim, self.layer_dim)
        self.layer_6 = nn.Linear(self.layer_dim, self.layer_dim)
        self.layer_7 = nn.Linear(self.layer_dim, self.layer_dim // 2)
        self.layer_8 = nn.Linear(self.layer_dim // 2, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        feat = self.activation(self.layer_0(x))
        feat = self.activation(self.layer_1(feat))
        feat = self.activation(self.layer_2(feat))
        feat = self.activation(self.layer_3(feat))
        feat = self.activation(self.layer_4(feat))
        feat = self.activation(self.layer_5(torch.cat((x, feat), dim=-1)))
        feat = self.activation(self.layer_6(feat))
        feat = self.activation(self.layer_7(feat))
        fluor = self.layer_8(feat)

        return fluor