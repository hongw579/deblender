import tensorflow as tf
from model import RDN
from utils_data import DataSampler
from classifier import classifier_session
import numpy as np
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from plotting import make_plot_full_blend2_compact_rescale
import math
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", False, "if the train")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")  ##
flags.DEFINE_integer("image_size", 80, "the size of image input")
flags.DEFINE_integer("c_dim", 3, "the size of channel")
flags.DEFINE_integer("scale", 1, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "the size of stride")  ##
flags.DEFINE_integer("epoch", 200, "number of epoch")
flags.DEFINE_integer("batch_size", 1, "the size of batch")
flags.DEFINE_float("learning_rate", 1e-4 , "the learning rate")
flags.DEFINE_float("lr_decay_steps", 10 , "steps of learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.5 , "rate of learning rate decay")
flags.DEFINE_boolean("is_eval", False, "if the evaluation")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_string("checkpoint_dir", "ckpt_rdn/", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "./data/prediction", "name of the result directory")
flags.DEFINE_string("train_set", "DIV2K_train_HR", "name of the train set")
flags.DEFINE_string("test_set", "Set5", "name of the test set")
flags.DEFINE_integer("D", 16, "D")
flags.DEFINE_integer("C", 8, "C")
flags.DEFINE_integer("G", 64, "G")
flags.DEFINE_integer("G0", 64, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")

data_shapes = {
    'blended': (80, 80, 3),
    'x': (80, 80, 3),
    'y': (80, 80, 3)
}

sampler = DataSampler(
    train_path='train',
    valid_path='valid',
    test_path='test',
    data_shapes=data_shapes,
    batch_size=1
)

def main(_):

    g_rdn = tf.Graph()
    g_class = tf.Graph()

    sess_rdn = tf.Session(graph=g_rdn)
    sess_class = tf.Session(graph=g_class)

    with sess_rdn.as_default():
        with sess_rdn.graph.as_default():
            rdn = RDN(sess_rdn,
                      is_train = FLAGS.is_train,
                      is_eval = FLAGS.is_eval,
                      image_size = FLAGS.image_size,
                      c_dim = FLAGS.c_dim,
                      scale = FLAGS.scale,
                      batch_size = FLAGS.batch_size,
                      D = FLAGS.D,
                      C = FLAGS.C,
                      G = FLAGS.G,
                      G0 = FLAGS.G0,
                      kernel_size = FLAGS.kernel_size,
                      sampler = sampler
                      )

            rdn.load_sess_rdn(FLAGS)

    with sess_class.as_default():
        with sess_class.graph.as_default():
            m_classifier = classifier_session(sess_class, FLAGS.batch_size, ckptdir = 'ckpt_classifier', global_step = 'vgg-28125')
            m_classifier.load_sess_class()

    n_batch = 0
    high_q = 0
    medium_q =0
    low_q = 0
    low_q_prev = 0
    #idx_y = []
    psnrs_x, psnrs_y, ssims_x, ssims_y = [], [], [], []
    psnrs_x_high, psnrs_x_mid, psnrs_x_low = [], [], []
    psnrs_y_high, psnrs_y_mid, psnrs_y_low = [], [], []
    ssims_x_high, ssims_x_mid, ssims_x_low = [], [], []
    ssims_y_high, ssims_y_mid, ssims_y_low = [], [], []

    while True:
        try:
            deblend_iter = []
            seq_remain = []
            residual_iter = []
            seq_pred = []
            seq_true = []
            iteration = 0
            y_analysis = False
            with sess_rdn.as_default():
                with sess_rdn.graph.as_default():
                    n_batch += 1
                    blended, true_x, true_y = sess_rdn.run([rdn.blend_img, rdn.x_img, rdn.y_img])
                    gan_x = sess_rdn.run(rdn.pred, feed_dict={rdn.images: blended}) 
                    true_x, true_y, gan_x = map(lambda x: (x + 1)/2, [true_x, true_y, gan_x])

                    gan_x_clip = np.clip(gan_x, 0., 1.) 
                    gan_x_clip[:, :3, :, :] = 0.
                    gan_x_clip[:, :, :3, :] = 0.
                    gan_x_clip[:, -3:, :, :] = 0.
                    gan_x_clip[:, :, -3:, :] = 0.

                    residual_x = blended-gan_x_clip
                    residual_clip = np.clip(residual_x, 0., 1.)
                    residual_iter.append(blended[0])
                    deblend_iter.append(gan_x_clip[0])
                    psnrs_x.append(compare_psnr(im_true=true_x[0], im_test=gan_x_clip[0]))
                    ssims_x.append(compare_ssim(X=true_x[0], Y=gan_x_clip[0], multichannel=True))
            with sess_class.as_default():
                with sess_class.graph.as_default():
                    N_blended = sess_class.run(m_classifier.pred,  feed_dict={m_classifier.x_pl:blended, m_classifier.drop_rate:0.0})
                    seq_remain.append(N_blended[0])
                    N_true_x = sess_class.run(m_classifier.pred,  feed_dict={m_classifier.x_pl:true_x, m_classifier.drop_rate:0.0})
                    seq_true.append(N_true_x[0])
                    N_true_y = sess_class.run(m_classifier.pred,  feed_dict={m_classifier.x_pl:true_y, m_classifier.drop_rate:0.0})
                    seq_true.append(N_true_y[0])
            while True:
                iteration += 1
                with sess_class.as_default():
                    with sess_class.graph.as_default():
                        max_per_image = residual_clip.max(axis=-1).max(axis=-1).max(axis=-1)
                        residual_norm = residual_clip/max_per_image

                        N_pred = sess_class.run(m_classifier.pred,  feed_dict={m_classifier.x_pl:gan_x_clip, m_classifier.drop_rate:0.0})
                        seq_pred.append(N_pred[0])

                        N_remain = sess_class.run(m_classifier.pred,  feed_dict={m_classifier.x_pl:residual_clip, m_classifier.drop_rate:0.0})
                        seq_remain.append(N_remain[0])
                        residual_iter.append(residual_clip[0])
                if N_remain == 0:
                    break
                if len(seq_remain) >=3 and seq_remain[-3]==seq_remain[-2]==seq_remain[-1]:
                    break
                if iteration == 7:
                    break
                with sess_rdn.as_default():
                    with sess_rdn.graph.as_default():
                        gan_x = sess_rdn.run(rdn.pred, feed_dict={rdn.images: residual_norm})
                        gan_x = (gan_x + 1)/2 
                        gan_x = gan_x*max_per_image
                        gan_x_clip = np.clip(gan_x, 0., 1.) 
                        gan_x_clip[:, :3, :, :] = 0.
                        gan_x_clip[:, :, :3, :] = 0.
                        gan_x_clip[:, -3:, :, :] = 0.
                        gan_x_clip[:, :, -3:, :] = 0.
                        residual_x = residual_clip-gan_x_clip
                        residual_clip = np.clip(residual_x, 0., 1.)
                        deblend_iter.append(gan_x_clip[0])
                        if iteration == 1:
                            psnrs_y.append(compare_psnr(im_true=true_y[0], im_test=gan_x_clip[0]))
                            ssims_y.append(compare_ssim(X=true_y[0], Y=gan_x_clip[0], multichannel=True))
                            y_analysis = True
            if seq_remain[::-1] == list(range(3)): 
                high_q += 1
                psnrs_x_high.append(psnrs_x[-1])
                ssims_x_high.append(ssims_x[-1])
                if y_analysis:
                    psnrs_y_high.append(psnrs_y[-1])
                    ssims_y_high.append(ssims_y[-1])
            elif seq_remain[-1] == 0:
                medium_q += 1
                psnrs_x_mid.append(psnrs_x[-1])
                ssims_x_mid.append(ssims_x[-1])
                if y_analysis:
                    psnrs_y_mid.append(psnrs_y[-1])
                    ssims_y_mid.append(ssims_y[-1])
            else:
                low_q += 1
                psnrs_x_low.append(psnrs_x[-1])
                ssims_x_low.append(ssims_x[-1])
                if y_analysis:
                    psnrs_y_low.append(psnrs_y[-1])
                    ssims_y_low.append(ssims_y[-1])
            if n_batch<=40:
                make_plot_full_blend2_compact_rescale(deblend_iter, residual_iter, seq_remain, seq_pred, seq_true, true_x, true_y, FLAGS.result_dir, n_batch)
                
        except tf.errors.OutOfRangeError:
            break
    print('num_q:', high_q,medium_q, low_q, n_batch)
    print('psnr x y', len(psnrs_x), len(psnrs_y))
    print('ssim x y', len(ssims_x), len(ssims_y))
    psnrs_x = np.array(psnrs_x)
    psnrs_y = np.array(psnrs_y)
    ssims_x = np.array(ssims_x)
    ssims_y = np.array(ssims_y)
    print('psnrs_x', np.mean(psnrs_x), np.median(psnrs_x))
    print('psnrs_y', np.mean(psnrs_y), np.median(psnrs_y))
    print('ssims_x', np.mean(ssims_x), np.median(ssims_x))
    print('ssims_y', np.mean(ssims_y), np.median(ssims_y))
    print('psnr x high mid low', len(psnrs_x_high), len(psnrs_y_high), len(psnrs_x_mid), len(psnrs_y_mid), len(psnrs_x_low), len(psnrs_y_low))
    print('ssim x high mid low', len(ssims_x_high), len(ssims_y_high), len(ssims_x_mid), len(ssims_y_mid), len(ssims_x_low), len(ssims_y_low))
    psnrs_x_high = np.array(psnrs_x_high)
    psnrs_y_high = np.array(psnrs_y_high)
    ssims_x_high = np.array(ssims_x_high)
    ssims_y_high = np.array(ssims_y_high)
    psnrs_x_mid = np.array(psnrs_x_mid)
    psnrs_y_mid = np.array(psnrs_y_mid)
    ssims_x_mid = np.array(ssims_x_mid)
    ssims_y_mid = np.array(ssims_y_mid)
    psnrs_x_low = np.array(psnrs_x_low)
    psnrs_y_low = np.array(psnrs_y_low)
    ssims_x_low = np.array(ssims_x_low)
    ssims_y_low = np.array(ssims_y_low)
    print('psnrs_x_high', np.mean(psnrs_x_high), np.median(psnrs_x_high))
    print('psnrs_y_high', np.mean(psnrs_y_high), np.median(psnrs_y_high))
    print('ssims_x_high', np.mean(ssims_x_high), np.median(ssims_x_high))
    print('ssims_y_high', np.mean(ssims_y_high), np.median(ssims_y_high))
    print('psnrs_x_mid', np.mean(psnrs_x_mid), np.median(psnrs_x_mid))
    print('psnrs_y_mid', np.mean(psnrs_y_mid), np.median(psnrs_y_mid))
    print('ssims_x_mid', np.mean(ssims_x_mid), np.median(ssims_x_mid))
    print('ssims_y_mid', np.mean(ssims_y_mid), np.median(ssims_y_mid))
    print('psnrs_x_low', np.mean(psnrs_x_low), np.median(psnrs_x_low))
    print('psnrs_y_low', np.mean(psnrs_y_low), np.median(psnrs_y_low))
    print('ssims_x_low', np.mean(ssims_x_low), np.median(ssims_x_low))
    print('ssims_y_low', np.mean(ssims_y_low), np.median(ssims_y_low))
if __name__=='__main__':
    tf.app.run()
