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

    def flow(self, X, y=None, batch_size=32, super_batch_size=10000, shuffle=True, seed=None):
        return HDF5Iterator3(X, y, self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            seed=seed,
                            dim_ordering=self.dim_ordering)

    def get_num_samples(self, X, y=None, return_shapes=False):
        """
        This method gives the number of samples.
        Useful for `samples_per_epoch` and `nb_val_samples` in `model.fit_generator()`.

        :param X: name of the dataset in the HDF5 file that represents the features (inputs)
        :param y: name of the dataset that represents the labels (targets). If None, only take into account X shape
        :return: number of samples for X (and y) in the HDF5 file
        """

        # Open provided HDF file
        h5f = h5py.File(self.hdf5_file, 'r')

        # if X is group in this HDF5 file
        if X in h5f:
            x_shape = h5f[X].shape
        else:
            raise ValueError('Provided key for features does not exist in this HDF5 file.'
                             'Available keys are : {}\n'.format(h5f.keys()))

        if y is not None:
            if y in h5f:
                y_shape = h5f[y].shape
            else:
                raise ValueError('Provided key for labels does not exist in this HDF5 file.'
                                 'Available keys are : {}\n'.format(h5f.keys()))

            if y_shape[0] != x_shape[0]:
                raise ValueError('Features and labels datasets don\'t have the same length'
                                 'X.shape : {}  and y.shape : {}'.format(x_shape, y_shape))
        else:
            # y is None
            y_shape = None

        # Now we can close the HDF5 file
        h5f.close()

        if return_shapes:
            return x_shape[0], x_shape, y_shape
        else:
            return x_shape[0]


class HDF5Iterator(object):

    def __init__(self, X, y, hdf5_generator, batch_size=32, shuffle=True, seed=None, dim_ordering='default'):

        self.X = X
        self.y = y
        self.hdf5_generator = hdf5_generator

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        self.batch_size = batch_size

        self.n, self.x_shape, self.y_shape = hdf5_generator.get_num_samples(X, y, return_shapes=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(self.n, batch_size, shuffle, seed)

        h5f = h5py.File(self.hdf5_generator.hdf5_file, 'r')
        self.dset_x = h5f[self.X]

        if self.y is not None:
            self.dset_y = h5f[self.y]
        else:
            self.dset_y = None

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

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/

        # open HDF5 file

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        sorted_index = sorted(index_array)
        batch_x = self.dset_x[sorted_index].astype(K.floatx())

        if self.y is None:

            return batch_x

        batch_y = self.dset_y[sorted_index]

        return batch_x, batch_y


class HDF5Iterator2(object):

    def __init__(self, X, y, hdf5_generator, super_batch_size=10000, batch_size=32, shuffle=True,
                 seed=None, dim_ordering='default'):

        self.X = y
        self.y = y
        self.hdf5_generator = hdf5_generator

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        self.super_batch_size = super_batch_size
        self.batch_size = batch_size

        self.n, self.x_shape, self.y_shape = hdf5_generator.get_num_samples(X, y, return_shapes=True)
        self.shuffle = shuffle

        self.super_batch_index = 0
        self.total_super_batches_seen = 0
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(self.n, super_batch_size, batch_size, shuffle, seed)

        # open HDF5 file
        self.h5f = h5py.File(hdf5_generator.hdf5_file, 'r')
        self.dset_x = self.h5f[X]

        if y is not None:
             self.dset_y = self.h5f[y]
        else:
             self.dset_y = None

        # super batch
        self.super_batch_x = None
        self.super_batch_y = None

    def reset_super_batch(self):
        self.super_batch_index = 0

    def reset_batch(self):
        self.batch_index = 0

    def _flow_index(self, n, super_batch_size=200, batch_size=32, shuffle=True, seed=None):
        # ensure self.batch_index is
        self.reset_super_batch()
        self.reset_batch()

        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            if self.super_batch_index == 0:
                index_array_super_batch = np.arange(n)
                if shuffle:
                    index_array_super_batch = np.random.permutation(n)

            if self.batch_index == 0:

                current_super_batch_index = (self.super_batch_index*super_batch_size) % n

                if n >= current_super_batch_index + super_batch_size:
                    current_super_batch_size = super_batch_size
                    self.super_batch_index += 1
                else:
                    current_super_batch_size = n - current_super_batch_index
                    self.super_batch_index = 0

                self.total_super_batches_seen += 1

                self.super_batch_x = self.dset_x[sorted(index_array_super_batch[current_super_batch_index:
                                                        current_super_batch_index + current_super_batch_size:])]
                if self.y is not None:
                    self.super_batch_y = self.dset_y[sorted(index_array_super_batch[current_super_batch_index:
                                                     current_super_batch_index + current_super_batch_size:])]

                index_array = np.arange(current_super_batch_size)

            current_index = (self.batch_index * batch_size) % current_super_batch_size

            if current_super_batch_size >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = current_super_batch_size - current_index
                self.batch_index = 0
            self.total_batches_seen += 1

            # print("Generate batch with size={}, index={}, from super_batch index={} with size={}".
            # format(current_batch_size,
            #               self.batch_index,
            #               self.super_batch_index,
            #               current_super_batch_size))
            # # time.sleep(1)

            yield (index_array[current_index:current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = self.super_batch_x[index_array]

        if self.y is None:
            return batch_x

        batch_y = self.super_batch_y[index_array]

        return batch_x, batch_y


class HDF5Iterator3(object):
    def __init__(self, X, y, hdf5_generator, batch_size=32, shuffle=True, seed=None, dim_ordering='default'):

        self.X = X
        self.y = y
        self.hdf5_generator = hdf5_generator

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        self.batch_size = batch_size

        self.n, self.x_shape, self.y_shape = hdf5_generator.get_num_samples(X, y, return_shapes=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(self.n, batch_size, shuffle, seed)

        h5f = h5py.File(self.hdf5_generator.hdf5_file, 'r')
        self.dset_x = h5f[self.X]

        if self.y is not None:
            self.dset_y = h5f[self.y]
        else:
            self.dset_y = None

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
            return (index_array[current_index:current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/

        # open HDF5 file

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        sorted_index = sorted(index_array)
        # batch_x = self.dset_x[sorted_index].astype(K.floatx())
        batch_x = compute([delayed(self.dset_x.__getitem__)(i) for i in sorted_index], get=threaded.get)

        if self.y is None:
            yield batch_x

        batch_y = compute([delayed(self.dset_y.__getitem__)(i) for i in sorted_index], get=threaded.get)

        yield batch_x, batch_y


class HDF5Dataset(object):
    def __init__(self, file, x, y=None):
        if os.path.exists(file):
            self.hdf5_file = h5py.File(file, 'r')
        else:
            raise ValueError('File nor found')

        self.num_samples = self.get_num_samples(x, y)
        self.x = x
        self.y = y

    def get_num_samples(self, x, y=None, return_shapes=False):

        # if X is group in this HDF5 file
        if x in self.hdf5_file:
            x_shape = self.hdf5_file[x].shape
        else:
            raise ValueError('Provided key for features does not exist in this HDF5 file.'
                             'Available keys are : {}\n'.format(self.hdf5_file.keys()))

        if y is not None:
            if y in self.hdf5_file:
                y_shape = self.hdf5_file[y].shape
            else:
                raise ValueError('Provided key for labels does not exist in this HDF5 file.'
                                 'Available keys are : {}\n'.format(self.hdf5_file.keys()))

            if y_shape[0] != x_shape[0]:
                raise ValueError('Features and labels datasets don\'t have the same length'
                                 'X.shape : {}  and y.shape : {}'.format(x_shape, y_shape))
        else:
            # y is None
            y_shape = None


        if return_shapes:
            return x_shape[0], x_shape, y_shape
        else:
            return x_shape[0]

    def batch_generator(self, batch_size=32, shuffle=True):
        num_batches_per_epoch = int((self.num_samples - 1) / batch_size) + 1

        x_dset = self.hdf5_file[self.x]
        if self.y is not None:
            y_dset = self.hdf5_file[self.y]
        else:
            y_dset = None

        while 1:
            indices = np.arange(self.num_samples)
            if shuffle:
                indices = np.random.permutation(indices)

            for num_batch in range(num_batches_per_epoch):
                start_index = num_batch*batch_size
                end_index = min((num_batch+1)*batch_size, self.num_samples)
                if shuffle:
                    batch_indices = sorted(list(indices[start_index:end_index]))
                else:
                    batch_indices = list(indices[start_index:end_index])

                if y_dset is None:
                    yield compute([delayed(x_dset.__getitem__)(i) for i in batch_indices], get=threaded.get)
                else:
                    yield compute([delayed(x_dset.__getitem__)(i) for i in batch_indices], get=threaded.get), \
                          compute([delayed(y_dset.__getitem__)(i) for i in batch_indices], get=threaded.get)
