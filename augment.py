import torch
import logging
import torch.nn as nn
import numpy as np


class TimeShift(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            shift = torch.empty(1).normal_(self.mean, self.std).int().item()
            x = torch.roll(x, shift, dims=0)
        return x




class TimeMask(nn.Module):
    def __init__(self, n=1, p=50):
        super().__init__()
        self.p = p
        self.n = n

    def forward(self, x):
        time, freq = x.shape
        if self.training:
            for i in range(self.n):
                t = torch.empty(1, dtype=int).random_(self.p).item()
                to_sample = max(time - t, 1)
                t0 = torch.empty(1, dtype=int).random_(to_sample).item()
                x[t0:t0 + t, :] = 0
        return x


class FreqMask(nn.Module):
    def __init__(self, n=1, p=12):
        super().__init__()
        self.p = p
        self.n = n

    def forward(self, x):
        time, freq = x.shape
        if self.training:
            for i in range(self.n):
                f = torch.empty(1, dtype=int).random_(self.p).item()
                f0 = torch.empty(1, dtype=int).random_(freq - f).item()
                x[:, f0:f0 + f] = 0.
        return x
