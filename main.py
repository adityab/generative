import os
import scipy.misc
import numpy as np
import tensorflow as tf
from utils import pp, visualize, to_json

from models.dcgan_mnist import DCGAN_MNIST
from models.dcgan_celeba import DCGAN_CelebA
from models.dcwgan_mnist import DCWGAN_MNIST
from models.dcwgan_celeba import DCWGAN_CelebA
from models.bigan_celeba import BiGAN_CelebA
from models.biwgan_celeba import BiWGAN_CelebA

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("stash_dir", ".", "The directory where all generated content goes")
flags.DEFINE_string("model", "dcgan_celeba", "The name of model to use [dcgan_celeba, dcgan_mnist, dcwgan_mnist]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", FLAGS.stash_dir + "/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", FLAGS.stash_dir + "/samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.model == 'dcgan_mnist':
      FLAGS.dataset = 'mnist'
      model = DCGAN_MNIST(
          sess,
          batch_size=FLAGS.batch_size,
          y_dim=10,
          z_dim=100,
          c_dim=1,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    elif FLAGS.model == 'dcgan_celeba':
      FLAGS.dataset = 'celebA'
      model = DCGAN_CelebA(
          sess,
          batch_size=FLAGS.batch_size,
          c_dim=FLAGS.c_dim,
          input_fname_pattern=FLAGS.input_fname_pattern,
          is_crop=FLAGS.is_crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    elif FLAGS.model == 'dcwgan_mnist':
      FLAGS.dataset = 'mnist'
      model = DCWGAN_MNIST(
          sess,
          batch_size=FLAGS.batch_size,
          n_critic=5,
          y_dim=10,
          z_dim=100,
          c_dim=1,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    elif FLAGS.model == 'dcwgan_celeba':
      FLAGS.dataset = 'celebA'
      model = DCWGAN_CelebA(
          sess,
          batch_size=FLAGS.batch_size,
          n_critic=5,
          c_dim=FLAGS.c_dim,
          input_fname_pattern=FLAGS.input_fname_pattern,
          is_crop=FLAGS.is_crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    elif FLAGS.model == 'bigan_celeba':
      FLAGS.dataset = 'celebA'
      model = BiGAN_CelebA(
          sess,
          batch_size=FLAGS.batch_size,
          c_dim=FLAGS.c_dim,
          input_fname_pattern=FLAGS.input_fname_pattern,
          is_crop=FLAGS.is_crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    elif FLAGS.model == 'biwgan_celeba':
      FLAGS.dataset = 'celebA'
      model = BiWGAN_CelebA(
          sess,
          batch_size=FLAGS.batch_size,
          c_dim=FLAGS.c_dim,
          input_fname_pattern=FLAGS.input_fname_pattern,
          is_crop=FLAGS.is_crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)

    else:
      print('No such model.')

    if FLAGS.is_train:
      model.train(FLAGS)
    else:
      if not model.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    OPTION = 1
    visualize(sess, model, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
