import tensorflow as tf
from vgg16_pre import VGG16
import numpy as np
import os
#from plotting import make_plot

class classifier_session(object):
    def __init__(self, sess, batch_size, n_cls=4, ckptdir='./log/ckpt/', savedir='./log/pred/', global_step = 'checkpoint-2000'):
        self.sess = sess
        self.batch_size = batch_size
        self.n_cls = n_cls
        self.ckptdir = ckptdir
        self.savedir = savedir
        self.global_step = global_step

        assert os.path.isdir(self.ckptdir)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
    
    def _parse_function(self, example_proto):
        keys_to_features = {'image_raw':tf.FixedLenFeature([19200], tf.float32),
                            'label': tf.FixedLenFeature([1], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['image_raw'], parsed_features['label']
    
    def load_sess_class(self):
        tfrecord_file_path = '../data_classifier_x_y_gan/'
        self.x_pl = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 3], name='input')
        self.y_pl = tf.placeholder(dtype=tf.float32, shape=[None, self.n_cls], name='label')
        self.drop_rate = tf.placeholder(tf.float32)

        self.model = VGG16(x = self.x_pl, drop_rate = self.drop_rate, num_classes = self.n_cls)
        self.output = self.model.fc8
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_pl))
        self.pred = tf.argmax(self.output,1)
        self.correct = tf.cast(tf.equal(tf.argmax(self.output,1), tf.argmax(self.y_pl, 1)), tf.float32)
        self.incorrect = tf.cast(tf.not_equal(tf.argmax(self.output,1), tf.argmax(self.y_pl, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=3)

        self.sess.run(init)
        self.ckpt_path = os.path.join(self.ckptdir, self.global_step)
        print('restoring from ', self.ckpt_path)
        self.saver.restore(self.sess, self.ckpt_path)
        
if __name__ == '__main__':
    test()
