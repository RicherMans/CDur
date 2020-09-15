from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from h5py import File
from tqdm import tqdm
import torch.utils.data as tdata


class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 labels: pd.DataFrame,
                 transform=None,
                 colname=('filename', 'encoded')):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self.dataset = None
        self._labels = labels
        self._colname = colname
        self._len = len(self._labels[colname[0]])
        # IF none is passed still use no transform at all
        self._transform = transform
        with File(self._h5file, 'r') as store:
            fname = self._labels[colname[0]][0]
            self.datadim = store[str(fname)].shape[-1]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = File(self._h5file, 'r', libver='latest')
        fname = self._labels[self._colname[0]][index]
        target = self._labels[self._colname[1]][index]
        if scipy.sparse.issparse(target):
            target = target.toarray().squeeze(0)
        target = target.tolist()
        data = self.dataset[fname][()]
        data = torch.as_tensor(data).float()
        if self._transform:
            data = self._transform(data)
        return data, target, fname


class MinimumOccupancySampler(tdata.Sampler):
    """
        docstring for MinimumOccupancySampler
        samples at least one instance from each class sequentially
    """
    def __init__(self, labels, sampling_mode='same', random_state=None):
        self.labels = labels
        data_samples, n_labels = labels.shape
        label_to_idx_list, label_to_length = [], []
        self.random_state = np.random.RandomState(seed=random_state)
        for lb_idx in range(n_labels):
            label_selection = labels[:, lb_idx]
            if scipy.sparse.issparse(label_selection):
                label_selection = label_selection.toarray()
            label_indexes = np.where(label_selection == 1)[0]
            self.random_state.shuffle(label_indexes)
            label_to_length.append(len(label_indexes))
            label_to_idx_list.append(label_indexes)

        self.longest_seq = max(label_to_length)
        self.data_source = np.zeros((self.longest_seq, len(label_to_length)),
                                    dtype=np.uint32)
        # Each column represents one "single instance per class" data piece
        for ix, leng in enumerate(label_to_length):
            # Fill first only "real" samples
            self.data_source[:leng, ix] = label_to_idx_list[ix]

        self.label_to_idx_list = label_to_idx_list
        self.label_to_length = label_to_length

        if sampling_mode == 'same':
            self.data_length = data_samples
        elif sampling_mode == 'over':  # Sample all items
            self.data_length = np.prod(self.data_source.shape)

    def _reshuffle(self):
        # Reshuffle
        for ix, leng in enumerate(self.label_to_length):
            leftover = self.longest_seq - leng
            random_idxs = self.random_state.randint(leng, size=leftover)
            self.data_source[leng:,
                             ix] = self.label_to_idx_list[ix][random_idxs]

    def __iter__(self):
        # Before each epoch, reshuffle random indicies
        self._reshuffle()
        n_samples = len(self.data_source)
        random_indices = self.random_state.permutation(n_samples)
        data = np.concatenate(
            self.data_source[random_indices])[:self.data_length]
        return iter(data)

    def __len__(self):
        return self.data_length


class MultiBalancedSampler(tdata.sampler.Sampler):
    """docstring for BalancedSampler
    Samples for Multi-label training
    Sampling is not totally equal, but aims to be roughtly equal
    """
    def __init__(self, Y, replacement=False, num_samples=None):
        assert Y.ndim == 2, "Y needs to be one hot encoded"
        if scipy.sparse.issparse(Y):
            raise ValueError("Not supporting sparse amtrices yet")
        class_counts = np.sum(Y, axis=0)
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum()
        classes = np.arange(Y[0].shape[0])
        # Revert from many_hot to one
        class_ids = [tuple(classes.compress(idx)) for idx in Y]

        sample_weights = []
        for i in range(len(Y)):
            # Multiple classes were chosen, calculate average probability
            weight = class_weights[np.array(class_ids[i])]
            # Take the mean of the multiple classes and set as weight
            weight = np.mean(weight)
            sample_weights.append(weight)
        self._weights = torch.as_tensor(sample_weights, dtype=torch.float)
        self._len = num_samples if num_samples else len(Y)
        self._replacement = replacement

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(
            torch.multinomial(self._weights, self._len,
                              self._replacement).tolist())


def getdataloader(data_frame,
                  data_file,
                  transform=None,
                  colname=None,
                  **dataloader_kwargs):

    if colname is None:
        colname = ('filename', 'encoded')
    dset = HDF5Dataset(data_file,
                        data_frame,
                        colname=colname,
                        transform=transform)

    return tdata.DataLoader(dset,
                            collate_fn=sequential_collate,
                            **dataloader_kwargs)


def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    return padded_seq


def sequential_collate(batches):
    # sort length wise
    # batches.sort(key=lambda x: len(x), reverse=True)
    seqs = []
    for data_seq in zip(*batches):
        if isinstance(data_seq[0],
                      (torch.Tensor, np.ndarray)):  # is tensor, then pad
            data_seq = pad(data_seq)
        elif type(data_seq[0]) is list or type(
                data_seq[0]) is tuple:  # is label or something, do not pad
            data_seq = torch.as_tensor(data_seq)
        seqs.append(data_seq)
    return seqs
