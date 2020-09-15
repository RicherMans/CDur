import ignite.metrics as metrics
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()

    def forward(self, clip_prob, frame_prob, tar):
        return nn.functional.binary_cross_entropy(input=clip_prob, target=tar)

class BCELossWithLabelSmoothing(nn.Module):
    """docstring for BCELoss"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, frame_prob, tar):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            tar = tar * (1 - self.label_smoothing) + (
                1 - tar) * self.label_smoothing / (n_classes - 1)
        return nn.functional.binary_cross_entropy(clip_prob, tar)


# Reimplement Loss, because ignite loss only takes 2 args, not 3 and nees to parse kwargs around ... just *output does the trick
class Loss(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)

    def update(self, output):
        average_loss = self._loss_fn(*output)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

