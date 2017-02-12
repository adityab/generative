from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class BiWGAN_CelebA(object):
  def __init__(self, sess, n_critic=5, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='celebA',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.n_critic = n_critic
    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')
    self.d_bn5 = batch_norm(name='d_bn5')

    self.e_bn1 = batch_norm(name='e_bn1')
    self.e_bn2 = batch_norm(name='e_bn2')
    self.e_bn3 = batch_norm(name='e_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def discriminator(self, image, z, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      
      h0z = lrelu(self.d_bn4(linear(z, 1024, 'd_h0z_lin')))
      h1z = lrelu(self.d_bn5(linear(h0z, 1024, 'd_h1z_lin')))

      h = tf.concat_v2([h4, h1z], 1)
      out = linear(h, 1, 'd_out_lin')

      return out

  def encoder(self, image, y=None, reuse=False):
    with tf.variable_scope("encoder") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.gf_dim, name='e_h0_conv'))
      h1 = lrelu(self.e_bn1(conv2d(h0, self.gf_dim*2, name='e_h1_conv')))
      h2 = lrelu(self.e_bn2(conv2d(h1, self.gf_dim*4, name='e_h2_conv')))
      h3 = lrelu(self.e_bn3(conv2d(h2, self.gf_dim*8, name='e_h3_conv')))
      reshaped = tf.reshape(h3, [self.batch_size, -1])

      mu = linear(reshaped, self.z_dim, 'e_mu_lin')
      sig = linear(reshaped, self.z_dim, 'e_log_sig_sq')
      z = mu + tf.exp(sig / 2) * tf.random_normal(shape=tf.shape(mu))

      return z

  def generator(self, z, y=None, reuse=False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4, s_h8, s_h16 = \
          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
      s_w2, s_w4, s_w8, s_w16 = \
          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0))

      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1))

      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)

  def build_model(self):
    # =================== PIPELINE =====================
    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_height, self.c_dim]

    # Real and Fake image inputs
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs
    
    # Latent space vector
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    # Now compose GAN
    self.E = self.encoder(inputs)
    self.G = self.generator(self.z)
    self.D = self.discriminator(inputs, self.E)
    self.D_= self.discriminator(self.G, self.z, reuse=True)

    # Sampler
    self.AE = self.generator(self.encoder(inputs, reuse=True), reuse=True)

    # =================== LOSSES =======================
    # Losses for Real input, Fake input, and Generator loss
    self.d_loss_real = tf.reduce_mean(self.D)
    self.d_loss_fake = tf.reduce_mean(self.D_)
    self.g_loss = tf.reduce_mean(self.D_)
    self.e_loss = tf.reduce_mean(self.D)
    # Discriminator loss is the sum of these two losses
    self.d_loss = self.d_loss_real - self.d_loss_fake

    # =================== ANALYTICS ====================
    # Define Summaries
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)
    
    # Summarize Real and Fake Losses
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    # Summarize D and G losses
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.e_vars = [var for var in t_vars if 'e_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    """Train DCGAN"""
    data = glob(os.path.join(config.stash_dir, config.dataset, self.input_fname_pattern))
    #np.random.shuffle(data)
    
    # Use RMS Optimizers
    # d_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
    #           .minimize(self.d_loss, var_list=self.d_vars)
    # g_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
    #           .minimize(self.g_loss, var_list=self.g_vars)
    # e_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
    #           .minimize(self.e_loss, var_list=self.e_vars)

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    e_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.e_loss, var_list=self.e_vars)

    # Set up weight clipping
    clip_ops = []
    for var in self.d_vars:
      clip_ops.append(tf.assign(var, tf.clip_by_value(var, -0.05, 0.05)))

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    # Set up logging
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter(config.stash_dir + "/logs", self.sess.graph)

    # Sample Latent
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    # Sample Reals from dataset
    sample_files = data[0:self.sample_num]
    sample = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  is_crop=self.is_crop,
                  is_grayscale=self.is_grayscale) for sample_file in sample_files]
    if (self.is_grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)

    save_images(sample_inputs, [8, 8],
          '{}/inputs.png'.format(config.sample_dir))

    counter = 0
    start_time = time.time()

    # Try to load model from checkpoint
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in range(config.epoch):
      data = np.array(glob(os.path.join(
        config.stash_dir, config.dataset, self.input_fname_pattern)))
      batch_idxs = min(len(data), config.train_size) // config.batch_size

      for idx in range(0, batch_idxs):
        # Train critic for longer when the metric is not properly formed,
        # otherwise self.n_critic times per generator iteration
        if counter < 25 or np.mod(counter, 100) == 0:
          n_critic = 100
        else:
          n_critic = self.n_critic

        for t in range(0, n_critic):
          # Sample a batch from dataset
          shuffled_idxs = np.arange(len(data))
          np.random.shuffle(shuffled_idxs)
          shuffled_idxs_minibatch = shuffled_idxs[0:config.batch_size]
          # Crop images
          batch_files = data[shuffled_idxs_minibatch]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        is_crop=self.is_crop,
                        is_grayscale=self.is_grayscale) for batch_file in batch_files]
          if (self.is_grayscale):
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)
          # Sample a batch from latent space
          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
               .astype(np.float32)
          # Train D network
          _, __, d_loss, gradient = self.sess.run([d_optim, clip_ops, self.d_loss],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          #self.writer.add_summary(summary_str, counter)
          #print(np.array([np.array(gi) for gi in gradient]))

        # Sample a batch from latent space
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
             .astype(np.float32)

        # Train G and E networks
        _, __, g_loss, e_loss = self.sess.run([g_optim, e_optim, self.g_loss, self.e_loss],
          feed_dict={ self.inputs: batch_images, self.z: batch_z })
        #self.writer.add_summary(summary_str, counter)

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, d_loss, g_loss, e_loss))
        
        # After every 100th minibatch, generate fresh samples and save the image
        if np.mod(counter, 100) == 1:
          #try:
          recons, samples, d_loss, g_loss = self.sess.run(
            [self.AE, self.G, self.d_loss, self.g_loss],
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs,
            },
          )

          print('saving samples...')
          save_images(recons, [8, 8],
                '{}/train_{:02d}_{:04d}_recons.png'.format(config.sample_dir, epoch, idx))
          save_images(samples, [8, 8],
                '{}/train_{:02d}_{:04d}_samples.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "BiWGAN_CelebA.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
