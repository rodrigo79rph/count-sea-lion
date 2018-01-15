from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import threading
import h5py
from keras import backend as K
from dask import delayed, threaded, compute
import os


class HDF5Generator(object):
    def __init__(self, hdf5_file, dim_ordering='default'):
        self.hdf5_file = hdf5_file

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering

    def flow(self, x, y, mean=None, batch_size=32, shuffle=True, seed=None):
        return HDF5Iterator(x, y, self,
                            mean=None,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            seed=seed,
                            dim_ordering=self.dim_ordering)

    def get_num_samples(self, x, y, return_shapes=False):
        h5f = h5py.File(self.hdf5_file, 'r')

        # if X is group in this HDF5 file
        if x in h5f:
            x_shape = h5f[x].shape
        else:
            raise ValueError('Provided key for features does not exist in this HDF5 file.'
                             'Available keys are : {}\n'.format(h5f.keys()))

        if y in h5f:
            y_shape = h5f[y].shape
        else:
            raise ValueError('Provided key for labels does not exist in this HDF5 file.'
                             'Available keys are : {}\n'.format(h5f.keys()))

        if y_shape[0] != x_shape[0]:
            raise ValueError('Features and labels datasets don\'t have the same length'
                             'X.shape : {}  and y.shape : {}'.format(x_shape, y_shape))

        # Now we can close the HDF5 file
        h5f.close()

        if return_shapes:
            return x_shape[0], x_shape[1:], y_shape[1:]
        else:
            return x_shape[0]


class HDF5Iterator(object):

    def __init__(self, x, y, hdf5_generator, mean=None, batch_size=32, shuffle=True, seed=None, dim_ordering='default'):

        self.x = x
        self.y = y
        if mean is None:
            self.mean = np.zeros((3,), dtype=np.float32)
        else:
            self.mean = mean
            
        self.hdf5_generator = hdf5_generator

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        self.batch_size = batch_size

        self.n, self.x_shape, self.y_shape = hdf5_generator.get_num_samples(x, y, return_shapes=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(self.n, batch_size, shuffle, seed)

        self.h5f = h5py.File(self.hdf5_generator.hdf5_file, 'r')

        self.dset_x = self.h5f[self.x]
        self.dset_counts = self.h5f['counts']
        self.dset_y = self.h5f[self.y]

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=True, seed=None):
        # ensure self.batch_index is
        self.reset()

        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index:current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self, with_counts=False):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/

        # open HDF5 file

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        sorted_index = sorted(index_array)
        batch_x = self.dset_x[sorted_index].astype(K.floatx())
        batch_x /= 127.5
        batch_x -= 1.
        # batch_x[:, :, :, 0] -= self.mean[0]
        # batch_x[:, :, :, 1] -= self.mean[1]
        # batch_x[:, :, :, 2] -= self.mean[2]
        
        batch_y = self.dset_y[sorted_index].astype(K.floatx())

        if with_counts:
            counts = self.dset_counts[sorted_index]
            return batch_x, batch_y, counts

        return batch_x, batch_y

    def close(self):
        self.h5f.close()
