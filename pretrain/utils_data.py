import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DataSampler:
    """
    Creates TensorFlow Dataset objects from directories containing
    .tfrecord TensorFlow binaries and passes tensors to graph. The
    resulting sampler is reinitializable onto any of three datasets
    (training, validation, testing) via the initialize method.

    Args:
        train_path(str): training data filepath containing .tfrecords files.
        valid_path(str): validation data filepath containing .tfrecords files.
        test_path(str): test data filepath containing .tfrecords files.
        data_shapes(dict): data shape dictionary to specify reshaping operation.
        batch_size(int): number of samples per batch call.
        shuffle(bool): shuffle data (only applicable to training set).
        buffer_size(int): size of shuffled buffer TFDataset will draw from.

    Note: this class currently only supports float data. In the future, it will
    need to accomodate integer-valued data as well.
    """
    def __init__(self, train_path, valid_path, test_path, data_shapes,
        batch_size, shuffle=True, buffer_size=50000):
        assert isinstance(batch_size, int), "Batch size must be integer-valued."
        assert isinstance(buffer_size, int), "Buffer size must be integer-valued."

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.data_shapes = data_shapes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.initialized = False

    def initialize(self):
        valid, test = map(self.make_dataset, [self.valid_path, self.test_path])
        train = self.make_dataset(self.train_path, train=True)

        self.iter = tf.data.Iterator.from_structure(
            train.output_types, train.output_shapes)
        train_init, valid_init, test_init = map(
            self.iter.make_initializer, [train, valid, test])
        self.init_ops = dict(zip(['train', 'valid', 'test'],
            [train_init, valid_init, test_init]))
        self.initialized = True

    def make_dataset(self, filepath, train=False):
        files = [os.path.join(filepath, file) for file
            in os.listdir(filepath) if file.endswith('.tfrecords')]
        dataset = tf.data.TFRecordDataset(files).map(self.decoder)
        if train:
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=self.buffer_size)
            return dataset.repeat().batch(self.batch_size)
        else:
            return dataset.batch(self.batch_size)

    def get_dataset(self, dataset='train'):
        if not self.initialized:
            raise ValueError('Sampler must be initialized before dataset retrieval.')
        try:
            return self.init_ops.get(dataset)
        except:
            raise ValueError('Dataset unknown or unavailable.')

    def decoder(self, example_proto):
        feature_keys = {k: tf.FixedLenFeature(np.prod(v), tf.float32)
            for k, v in self.data_shapes.items()}
        parsed_features = tf.parse_single_example(example_proto, feature_keys)
        parsed = [parsed_features[key] for key in self.data_shapes.keys()]
        return parsed

    def get_batch(self):
        if not self.initialized:
            raise ValueError('Sampler must be initialized before batch retrieval.')
        batch = self.iter.get_next()
        batch = [tf.reshape(batch[i], [-1] + list(v))
            for i, v in enumerate(self.data_shapes.values())]
        print('batch and shape:', batch, len(batch), batch[0].shape)
        return batch
