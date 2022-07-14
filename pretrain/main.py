import tensorflow as tf
from model import RDN
from utils_data import DataSampler

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")  ##
flags.DEFINE_integer("image_size", 80, "the size of image input")
flags.DEFINE_integer("c_dim", 3, "the size of channel")
flags.DEFINE_integer("scale", 1, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "the size of stride")  ##
flags.DEFINE_integer("epoch", 200, "number of epoch")
flags.DEFINE_integer("batch_size", 16, "the size of batch")
flags.DEFINE_integer("N_GPU", 7, "num of gpus")
flags.DEFINE_float("learning_rate", 5e-4, "the learning rate")
flags.DEFINE_float("lr_decay_steps", 10 , "steps of learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.5 , "rate of learning rate decay")
flags.DEFINE_boolean("is_eval", False, "if the evaluation")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_string("checkpoint_dir", "ckptdir", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "result", "name of the result directory")
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
    batch_size=FLAGS.N_GPU*FLAGS.batch_size
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
#config.log_device_placement=True

def main(_):
    rdn = RDN(tf.Session(config=config),
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
              sampler = sampler,
              n_gpus = FLAGS.N_GPU
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)

if __name__=='__main__':
    tf.app.run()
