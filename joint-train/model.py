import tensorflow as tf
import numpy as np
from vgg16_pre import VGG16
import time
import os
from utils import average_gradients, edge_mask

class RDN(object):

    def __init__(self,
                 sess,
                 is_train,
                 is_eval,
                 image_size,
                 c_dim,
                 scale,
                 batch_size,
                 D,
                 C,
                 G,
                 G0,
                 kernel_size,
                 sampler,
                 n_gpus,
                 n_cls
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size
        self.data = sampler
        self.n_gpus=n_gpus
        self.n_cls=n_cls

    def SFEParams(self):
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1': tf.Variable(tf.random_normal([ks, ks, self.c_dim, G0], stddev=0.01), name='w_S_1'),
            'w_S_2': tf.Variable(tf.random_normal([ks, ks, G0, G], stddev=0.01), name='w_S_2')
        }
        biasesS = {
            'b_S_1': tf.Variable(tf.zeros([G0], name='b_S_1')),
            'b_S_2': tf.Variable(tf.zeros([G], name='b_S_2'))
        }

        return weightsS, biasesS

    def RDBParams(self):
        weightsR = {}
        biasesR = {}
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size

        for i in range(1, D+1):
            for j in range(1, C+1):
                weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=0.01), name='w_R_%d_%d' % (i, j))}) 
                biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
            weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=0.01), name='w_R_%d_%d' % (i, C+1))})
            biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

        return weightsR, biasesR

    def DFFParams(self):
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsD = {
            'w_D_1': tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev=0.01), name='w_D_1'),
            'w_D_2': tf.Variable(tf.random_normal([ks, ks, G0, G0], stddev=0.01), name='w_D_2')
        }
        biasesD = {
            'b_D_1': tf.Variable(tf.zeros([G0], name='b_D_1')),
            'b_D_2': tf.Variable(tf.zeros([G0], name='b_D_2'))
        }

        return weightsD, biasesD

    def UPNParams(self):
        G0 = self.G0
        weightsU = {
            'w_U_1': tf.Variable(tf.random_normal([5, 5, G0, 64], stddev=0.01), name='w_U_1'),
            'w_U_2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.01), name='w_U_2'),
            'w_U_3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w_U_3')
        }
        biasesU = {
            'b_U_1': tf.Variable(tf.zeros([64], name='b_U_1')),
            'b_U_2': tf.Variable(tf.zeros([32], name='b_U_2')),
            'b_U_3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b_U_3'))
        }

        return weightsU, biasesU

    def UPN(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.weightsU['w_U_1'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_1']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_2'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_2']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_3'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_3']

        x = self.PS(x, self.scale)

        return x

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D+1):
            x = rdb_in
            for j in range(1, self.C+1):
                tmp = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self.biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                x = tf.concat([x, tmp], axis=3)

            x = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i, self.C+1)], strides=[1,1,1,1], padding='SAME') +  self.biasesR['b_R_%d_%d' % (i, self.C+1)]
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)

    # NOTE: train with batch size 
    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (-1, a, b, r, r))
        X = tf.split(X, a, 1) 
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2) 
        X = tf.split(X, b, 1) 
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2) 
        return tf.reshape(X, (-1, a*r, b*r, 1))

    # NOTE: test without batchsize
    def _phase_shift_test(self, I ,r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1) 
        X = tf.concat([tf.squeeze(x) for x in X], 1) 
        X = tf.split(X, b, 0) 
        X = tf.concat([tf.squeeze(x) for x in X], 1) 
        return tf.reshape(X, (1, a*r, b*r, 1))

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) 
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) 
        return X

    def model(self, inputs):
        F_1 = tf.nn.conv2d(inputs, self.weightsS['w_S_1'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_1']
        F0 = tf.nn.conv2d(F_1, self.weightsS['w_S_2'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_2']

        FD = self.RDBs(F0)

        FGF1 = tf.nn.conv2d(FD, self.weightsD['w_D_1'], strides=[1,1,1,1], padding='SAME') + self.biasesD['b_D_1']
        FGF2 = tf.nn.conv2d(FGF1, self.weightsD['w_D_2'], strides=[1,1,1,1], padding='SAME') + self.biasesD['b_D_2']

        FDF = tf.add(FGF2, F_1)

        FU = self.UPN(FDF)
        IHR = tf.nn.conv2d(FU, self.weight_final, strides=[1,1,1,1], padding='SAME') + self.bias_final

        return IHR

    def build_model(self, images_shape, labels_shape):
        with tf.variable_scope('rdn', reuse=tf.AUTO_REUSE) as vs:
            self.weightsS, self.biasesS = self.SFEParams()
       	    self.weightsR, self.biasesR = self.RDBParams()
            self.weightsD, self.biasesD = self.DFFParams()
            self.weightsU, self.biasesU = self.UPNParams()
            self.weight_final = tf.Variable(tf.random_normal([self.kernel_size, self.kernel_size, self.c_dim, self.c_dim], stddev=np.sqrt(2.0/9/3)), name='w_f')
            self.bias_final = tf.Variable(tf.zeros([self.c_dim], name='b_f')),

    def train(self, config):
        print("\nPrepare Data...\n")
        self.data.initialize()        
        data_num = 50000
        batch_num = data_num // (config.batch_size*config.N_GPU)

        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size * self.scale, self.image_size * self.scale, self.c_dim]
        self.build_model(images_shape, labels_shape)

        classifier = VGG16(num_classes = self.n_cls)

        rdn_varlist = {v.op.name.lstrip("rdn/"): v
                    for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="rdn/")}
        rdn_saver = tf.train.Saver(var_list=rdn_varlist, max_to_keep=3)
        classifier_varlist = {v.op.name.lstrip("vgg16/"): v
                    for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="vgg16/")}
        classifier_saver = tf.train.Saver(var_list=classifier_varlist, max_to_keep=3)

        # First deblending of RDN
        self.blend_img, self.x_img, self.y_img = self.data.get_batch()

        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels_1 = tf.placeholder(tf.float32, labels_shape, name='labels_1')
        self.labels_2 = tf.placeholder(tf.float32, labels_shape, name='labels_2')

        self.model_name = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        counter = self.load(config.checkpoint_dir, restore=False)
        epoch_start = counter // batch_num
        batch_start = counter % batch_num

        global_step = tf.Variable(counter, trainable=False)
        lr_rdn = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps*batch_num, config.lr_decay_rate, staircase=True)
        lr_vgg = tf.Variable(1e-6, trainable=False)
        opt_rdn = tf.train.AdamOptimizer(learning_rate=lr_rdn)
        opt_to_vgg = tf.train.GradientDescentOptimizer(learning_rate=lr_vgg)
        opt_to_rdn = tf.train.GradientDescentOptimizer(learning_rate=lr_vgg)

        tower_grads_1 = []
        tower_loss_1 = []
        pred_rdn_1 = []
        for i_gpu in range(self.n_gpus):
            with tf.device("/gpu:%d" % i_gpu):
                _x_1 = self.images[i_gpu*self.batch_size: (i_gpu+1)*self.batch_size]
                _y_1 = self.labels_1[i_gpu*self.batch_size: (i_gpu+1)*self.batch_size]
                i_pred_1 = self.model(_x_1)
                pred_rdn_1.append(i_pred_1)
                i_loss_1 = tf.reduce_mean(tf.abs(_y_1 - i_pred_1))
                i_grads_1 = opt_rdn.compute_gradients(i_loss_1, self.rdn_vars)
                tower_grads_1.append(i_grads_1)
                tower_loss_1.append(i_loss_1)

        grads_1 = average_gradients(tower_grads_1)
        pred_rdn_1_all = tf.concat(pred_rdn_1, 0) 
        pred_rdn_1_p = (pred_rdn_1_all+1)/2
        pred_rdn_1_clip = tf.clip_by_value(pred_rdn_1_p, 0., 1.)
        mask_1 = edge_mask(config.batch_size*config.N_GPU)
        pred_rdn_1_clean = mask_1*pred_rdn_1_clip
        res_1 = self.images-pred_rdn_1_clean
        res_1_clip = tf.clip_by_value(res_1, 0., 1.)
        res_1_max = tf.reduce_max(res_1_clip, axis=[1,2,3], keepdims=True)
        res_1_max_value = tf.stop_gradient(res_1_max)
        res_1_norm = res_1_clip/res_1_max
        res_1_norm_value = tf.stop_gradient(res_1_norm)

        tower_loss_2 = []
        pred_rdn_2 = []
        for i_gpu in range(self.n_gpus):
            with tf.device("/gpu:%d" % i_gpu):
                _x_2 = res_1_norm_value[i_gpu*self.batch_size: (i_gpu+1)*self.batch_size]
                _y_2 = self.labels_2[i_gpu*self.batch_size: (i_gpu+1)*self.batch_size]
                i_res_1_max_value = res_1_max_value[i_gpu*self.batch_size: (i_gpu+1)*self.batch_size]
                i_pred_2 = self.model(_x_2)
                i_pred_2_bnorm = (i_pred_2+1)*i_res_1_max_value-1
                pred_rdn_2.append(i_pred_2_bnorm)
                i_loss_2 = tf.reduce_mean(tf.abs(_y_2 - i_pred_2_bnorm))
                tower_loss_2.append(i_loss_2) 

        learning_step_1 = opt_rdn.apply_gradients(grads_1)

        pred_rdn_2_all = tf.concat(pred_rdn_2, 0)
        pred_rdn_2_p = (pred_rdn_2_all+1)/2
        pred_rdn_2_clip = tf.clip_by_value(pred_rdn_2_p, 0., 1.)
        mask_2 = edge_mask(config.batch_size*config.N_GPU)
        pred_rdn_2_clean = mask_2*pred_rdn_2_clip
        res_2 = res_1_clip-pred_rdn_2_clean
        res_2_clip = tf.clip_by_value(res_2, 0., 1.)

        preblend1 = (self.labels_1+1)/2
        preblend2 = (self.labels_2+1)/2

        x_classifier_op = tf.concat([self.images, res_2_clip, res_1_clip, preblend1, preblend2, self.images, res_1_norm_value], 0)
        x_classifier = tf.stop_gradient(x_classifier_op)
        label0 = [0]*config.batch_size*config.N_GPU
        label1 = [1]*config.batch_size*config.N_GPU
        label2 = [2]*config.batch_size*config.N_GPU
        y_label0 = tf.one_hot(label0, self.n_cls)
        y_label1 = tf.one_hot(label1, self.n_cls)
        y_label2 = tf.one_hot(label2, self.n_cls)
        y_classifier_op = tf.concat([y_label2, y_label0, y_label1, y_label1, y_label1, y_label1, y_label1], 0)
        y_classifier = tf.stop_gradient(y_classifier_op)

        tower_grads_c = []
        tower_loss_c = []
        tower_grads_r = []
        tower_loss_r = []
        pred_c = []
        for i_gpu in range(7):
            with tf.device("/gpu:%d" % i_gpu):
                _x_c = x_classifier[i_gpu*self.batch_size*config.N_GPU: (i_gpu+1)*self.batch_size*config.N_GPU]
                _y_c = y_classifier[i_gpu*self.batch_size*config.N_GPU: (i_gpu+1)*self.batch_size*config.N_GPU]

                if i_gpu <= 1:
                    i_pred_c = classifier.model(_x_c)
                    pred_c.append(i_pred_c)
                    i_loss_c = 0.0001*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=i_pred_c, labels=_y_c))
                    i_grads_c = opt_to_vgg.compute_gradients(i_loss_c, classifier.vars)
                    tower_grads_c.append(i_grads_c)
                    tower_loss_c.append(i_loss_c)
                elif i_gpu >=2 and i_gpu <= 4:
                    i_pred_c = classifier.model(_x_c)
                    pred_c.append(i_pred_c)
                    i_loss_c = 0.0001*0.2*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=i_pred_c, labels=_y_c))
                    i_grads_c = opt_to_vgg.compute_gradients(i_loss_c, classifier.vars)
                    tower_grads_c.append(i_grads_c)
                    tower_loss_c.append(i_loss_c)
                elif i_gpu >= 5:
                    i_pred_r = self.model(_x_c)
                    r_pred_p = (i_pred_r+1)/2
                    if i_gpu == 6:
                        r_pred_p = r_pred_p*res_1_max_value
                    r_pred_clip = tf.clip_by_value(r_pred_p, 0., 1.)
                    mask_r = edge_mask(config.batch_size*config.N_GPU)
                    r_pred_clean = mask_r*r_pred_clip
                    i_pred_c = classifier.model(r_pred_clean)
                    pred_c.append(i_pred_c)
                    i_loss_r = 0.0001*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=i_pred_c, labels=_y_c))
                    i_loss_c = 0.2*i_loss_r
                    i_grads_c = opt_to_vgg.compute_gradients(i_loss_c, classifier.vars)
                    tower_grads_c.append(i_grads_c)
                    tower_loss_c.append(i_loss_c)
                    i_grads_r = opt_to_rdn.compute_gradients(i_loss_r, self.rdn_vars)
                    tower_grads_r.append(i_grads_r)
                    tower_loss_r.append(i_loss_r)

        pred_c_all = tf.concat(pred_c, 0)
        grads_c = average_gradients(tower_grads_c)
        grads_r = average_gradients(tower_grads_r)
        learning_step_vgg = opt_to_vgg.apply_gradients(grads_c)
        learning_step_rdn = opt_to_rdn.apply_gradients(grads_r)

        total_loss_c = tf.stack(tower_loss_c)
        loss_vgg = tf.reduce_mean(total_loss_c, 0)
        total_loss_r = tf.stack(tower_loss_r)
        loss_rdn = tf.reduce_mean(total_loss_r, 0)

        total_loss_1 = tf.stack(tower_loss_1)
        total_loss_2 = tf.stack(tower_loss_2)
        self.loss_1 = tf.reduce_mean(total_loss_1, 0) 
        self.loss_2 = tf.reduce_mean(total_loss_2, 0) 

        accuracy_vgg = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_c_all,1), tf.argmax(y_classifier, 1)), tf.float32))

        self.summary_loss1 = tf.summary.scalar('loss_1', self.loss_1)
        self.summary_loss2 = tf.summary.scalar('loss_2', self.loss_2)
        self.summary_image = tf.summary.image("Image_blended", self.images)
        self.summary_label1 = tf.summary.image("Image_X", self.labels_1)
        self.summary_label2 = tf.summary.image("Image_Y", self.labels_2)
        self.summary_image1 = tf.summary.image("Deblend_X", pred_rdn_1_clean)
        self.summary_image2 = tf.summary.image("Deblend_Y", pred_rdn_2_clean)

        image_summary_vgg = tf.summary.image("Image_vgg", x_classifier)
        loss_summary_vgg_3 = tf.summary.scalar("Loss_rdn", loss_rdn)
        loss_summary_vgg = tf.summary.scalar("Loss_vgg", loss_vgg)

        self.saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run(session=self.sess)
        rdn_saver.restore(self.sess, 'RDN.model-61800')
        classifier_saver.restore(self.sess, 'checkpoint-99999')

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter((os.path.join(config.checkpoint_dir, self.model_name, "log")), self.sess.graph)

        self.load(config.checkpoint_dir, restore=True)
        self.sess.run(self.data.get_dataset('train'))

        print("\nNow Start Training...\n")
        for ep in range(epoch_start, config.epoch):
            # Run by batch images
            for idx in range(batch_start, batch_num):
                counter += 1
                batch_blend, batch_x, batch_y = self.sess.run([self.blend_img, self.x_img, self.y_img])
                _, _, _, loss_x_np, loss_y_np, loss_vgg_np, loss_vgg_rdn_np, lr_rdn_np, lr_vgg_np, acc_vgg = self.sess.run([learning_step_1, learning_step_rdn, learning_step_vgg, self.loss_1, self.loss_2, loss_vgg, loss_rdn, lr_rdn, lr_vgg, accuracy_vgg], feed_dict={self.images: batch_blend, self.labels_1: batch_x, self.labels_2: batch_y})
                if counter % 20 == 0:
                    print("Epoch: [%4d], batch: [%6d/%6d], loss_x: [%.8f], loss_y: [%.8f], lr: [%.6f], step: [%d]" % ((ep+1), (idx+1), batch_num, loss_x_np, loss_y_np, lr_rdn_np, counter))
                    print("Classifier: Step [%d],  Loss : %f, Loss on RDN: %f, training accuracy : %f" % (counter, loss_vgg_np, loss_vgg_rdn_np, acc_vgg))
                if counter % 3300 == 0:
                    self.save(config.checkpoint_dir, counter)

                    if not os.path.isdir(os.path.join(config.checkpoint_dir, 'checkpoint/ckpt_rdn/rdn_16_8_64_x1/')):
                        os.makedirs(os.path.join(config.checkpoint_dir, 'checkpoint/ckpt_rdn/rdn_16_8_64_x1/'))
                    if not os.path.isdir(os.path.join(config.checkpoint_dir, 'checkpoint/ckpt_classifier/')):
                        os.makedirs(os.path.join(config.checkpoint_dir, 'checkpoint/ckpt_classifier/'))

                    rdn_saver.save(self.sess, save_path = config.checkpoint_dir + 'checkpoint/ckpt_rdn/rdn_16_8_64_x1/rdn', global_step=counter)
                    classifier_saver.save(self.sess, config.checkpoint_dir + 'checkpoint/ckpt_classifier/vgg', global_step=counter)

                    summary_str = self.sess.run(merged_summary_op, feed_dict={self.images: batch_blend, self.labels_1: batch_x, self.labels_2: batch_y})
                    summary_writer.add_summary(summary_str, counter)

                if counter > 0 and counter == batch_num * config.epoch:
                    self.save(config.checkpoint_dir, counter)
                    rdn_saver.save(self.sess, save_path = config.checkpoint_dir + 'checkpoint/ckpt_rdn/rdn_16_8_64_x1/rdn', global_step=counter)
                    classifier_saver.save(self.sess, config.checkpoint_dir + 'checkpoint/ckpt_classifier/vgg', global_step=counter)
                    break

        summary_writer.close()

    def eval(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)

        avg_time = 0
        avg_pasn = 0
        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            input_, label_ = get_image(paths[idx], config.scale, config.matlab_bicubic)

            images_shape = input_.shape
            labels_shape = label_.shape
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess) 

            self.load(config.checkpoint_dir, restore=True)

            time_ = time.time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time.time() - time_

            # import matlab.engine
            # eng = matlab.engine.start_matlab()
            # time_ = time.time()
            # result = np.asarray(eng.imresize(matlab.double((input_[0, :] / 255.0).tolist()), config.scale, 'bicubic'))
            # avg_time += time.time() - time_

            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_[0], config.scale)
            avg_pasn += psnr

            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time.time() - time_ , psnr))

            if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
                os.makedirs(os.path.join(os.getcwd(),config.result_dir))
            imsave(x[:, :, ::-1], config.result_dir + "/%d.png" % idx)

        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)

    def test(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)

        avg_time = 0
        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            input_ = imread(paths[idx])
            input_ = input_[:, :, ::-1]
            input_ = input_[np.newaxis, :]

            images_shape = input_.shape
            labels_shape = input_.shape * np.asarray([1, self.scale, self.scale, 1])
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess) 

            self.load(config.checkpoint_dir, restore=True)

            time_ = time.time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time.time() - time_

            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            x = x[:, :, ::-1]
            checkimage(np.uint8(x))

            if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
                os.makedirs(os.path.join(os.getcwd(),config.result_dir))
            imsave(x, config.result_dir + "/%d.png" % idx)

        print("Avg. Time:", avg_time / data_num)



    def load(self, checkpoint_dir, restore):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            step = int(os.path.basename(ckpt_path).split('-')[1])
            if restore:
                self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
                print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            if restore:
                print("\nCheckpoint Loading Failed! \n")

        return step

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "RDN.model"),
                        global_step=step)

    @property
    def rdn_vars(self):
        return [var for var in tf.global_variables() if 'rdn' in var.name]
