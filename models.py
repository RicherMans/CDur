from itertools import zip_longest
import numpy as np

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class ExtAttentionPool(nn.Module):
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.attention = nn.Linear(inputdim, outputdim)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.activ = nn.Softmax(dim=self.pooldim)

    def forward(self, logits, decision):
        # Logits of shape (B, T, D), decision of shape (B, T, C)
        w_x = self.activ(self.attention(logits) / self.outputdim)
        h = (logits.permute(0, 2, 1).contiguous().unsqueeze(-2) *
             w_x.unsqueeze(-1)).flatten(-2).contiguous()
        return torch.sum(h, self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


class MilSEDCNN(nn.Module):
    """

    Implemtation of "Adaptive pooling operators for weakly labeled
        sound event detection" https://arxiv.org/pdf/1804.10070.pdf

    Required input to be 128 dimensional since a fixed-dimensional pooling layer is used
    """
    def __init__(self, inputdim, outputdim, **kwargs):
        """__init__

        :param inputdim: Input dimension, can also be neglected
        :param outputdim:
        :param **kwargs:
        """
        super().__init__()
        assert inputdim == 128, "Only works for 128 dimensional mel"
        self._inputdim = inputdim
        self.network = nn.Sequential(
            nn.BatchNorm2d(1),
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # BLock 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=(1, 8), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256))

        def calculate_cnn_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]

        cnn_outputdim = calculate_cnn_size((1, 500, inputdim))
        linear_input_dim = cnn_outputdim[-1] * cnn_outputdim[0]
        # During training, pooling in time
        self.outputlayer = nn.Linear(linear_input_dim, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'soft'),
                                               inputdim=linear_input_dim,
                                               outputdim=outputdim)
        self.network.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        decision_time = torch.sigmoid(self.outputlayer(x))
        decision = self.temp_pool(x, decision_time).squeeze(1)
        decision = torch.clamp(decision, min=1e-7, max=1.)
        return decision, decision_time


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class cATP(nn.Module):
    """docstring for cATP"""
    def __init__(self, inputdim):
        super().__init__()
        self.inputdim = inputdim
        self.trans = nn.Linear(inputdim, 1)
        nn.init.zeros_(self.trans.weight)
        nn.init.zeros_(self.trans.bias)

    def forward(self, x):
        w_x = self.trans(x)
        # InputDim as a variable temperature
        a_ct = torch.softmax(w_x / self.inputdim, dim=1)
        h = torch.sum(x * a_ct, dim=1)
        return h, w_x


class cATPSDS(nn.Module):
    """ 
    Implementation of the paper:

    Specialized Decision Surface and Disentangled Feature for Weakly-Supervised Polyphonic Sound Event Detection

    https://arxiv.org/pdf/1905.10091.pdf

    """
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        assert outputdim == 10, "Just for DCASE2018/19, other datasets currently not supported"
        filters = [1] + kwargs.get('filters', [160, 160, 160])
        kernels = [5, 5, 3]
        paddings = [2, 2, 1]
        self.dimensions = kwargs.get('dimensions',
                                     [46, 22, 92, 42, 82, 17, 13, 160, 74, 85])
        self.outputdim = outputdim
        features = nn.ModuleList([nn.BatchNorm2d(1, eps=1e-4, momentum=0.01)])
        for h0, h1, kernel, padding in zip(filters, filters[1:], kernels,
                                           paddings):
            features.append(
                nn.Sequential(
                    nn.Conv2d(h0,
                              h1,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False),
                    nn.BatchNorm2d(h1, eps=1e-4, momentum=0.01), nn.ReLU(True),
                    nn.MaxPool2d((1, 4))))
        self.features = nn.Sequential(*features)
        init_weights(self.features)
        self.attentions = nn.ModuleList(
            [cATP(self.dimensions[f]) for f in range(outputdim)])
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.dimensions[f], 1) for f in range(outputdim)])

    def forward(self, x):
        """forward

        :param x: 3D input of shape B x T x D
        """
        x = x.unsqueeze(1)
        # B, 1, T, D
        x = self.features(x).flatten(-2).permute(0, 2, 1).contiguous()
        # B, T, C
        decision, decision_time = [], []
        for c in range(self.outputdim):
            sds = x[:, :, :self.dimensions[
                c]]  #self.dimensions[c] is the hyperparmeter `k` in the paper
            embedding_level, time_level = self.attentions[c](
                sds)  #Calcualte individual attention for the class
            decision.append(self.classifiers[c](embedding_level))
            decision_time.append(time_level)
        decision_time = torch.sigmoid(torch.cat(decision_time, dim=-1))
        decision = torch.sigmoid(torch.cat(decision, dim=-1)).squeeze(
            1)  # Remove time dimension
        decision = torch.clamp(decision, min=1e-7, max=1.)
        return decision, decision_time




class CDur(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        features = nn.ModuleList()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
                                               inputdim=256,
                                               outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        return decision, decision_time
