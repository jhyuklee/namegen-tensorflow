import tensorflow as tf
import os

from time import gmtime, strftime
from dataset import *
from run import train, train_vae
from model import NameGeneration
from vae import VAE


flags = tf.app.flags
flags.DEFINE_integer("vae_epoch", 300, "Epoch to train")
flags.DEFINE_integer("ae_epoch", 500, "Epoch to train")
flags.DEFINE_integer("gan_epoch", 10000, "Epoch to train")
flags.DEFINE_integer("input_dim", 43, "Data input dimension + PAD, GO, EOS")
flags.DEFINE_integer("class_dim", 127, "Data class dimension")
flags.DEFINE_integer("latent_dim", 30, "Latent variable dimenstion")
flags.DEFINE_integer("batch_size", 1000, "Mini-batch size")
flags.DEFINE_integer("max_time_step", 45, "Maximum time step of RNN")
flags.DEFINE_integer("min_grad", -10, "Minimum gradient to clip")
flags.DEFINE_integer("max_grad", 10, "Maximum gradient to clip")
flags.DEFINE_integer("cell_dim", 200, "Dimension of RNN cell")
flags.DEFINE_integer("cell_layer_num", 1, "The layer number of RNN ")
flags.DEFINE_integer("char_dim", 50, "Dimension of character embedding")
flags.DEFINE_integer("hidden_dim", 200, "Dimension of hidden layer for FFNN")
flags.DEFINE_float("ae_lr", 5e-3, "Learning rate of autoencoder")
flags.DEFINE_float("vae_lr", 1e-2, "Learning rate of variational autoencoder")
flags.DEFINE_float("cf_lr", 1e-3, "Learning rate of classifier")
flags.DEFINE_float("gan_lr", 1e-3, "Learning rate of GAN")
flags.DEFINE_float("output_dr", 0.5, "Dropout rate of FFNN")
flags.DEFINE_float("cell_keep_prob", 0.5, "Keep prob of RNN cell dropout")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
flags.DEFINE_string("results_dir", "results", "Directory name to save the results")
flags.DEFINE_string("data_dir", "data", "Directory name to save the results")
flags.DEFINE_boolean("conditional", False, "True to use conditional generation")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing")
flags.DEFINE_boolean("load_autoencoder", False, "True to load pretrained autoencoder")
flags.DEFINE_boolean("train_autoencoder", True, "True to train autoencoder")
flags.DEFINE_string("pretrained_path", "ae_no_class/pretrained_ae", "Path of pretrained ae")
FLAGS = flags.FLAGS


def create_model(config):
    scope = 'NameGeneration-' + strftime("%Y%m%d%H%M%S", gmtime())
    config.checkpoint_dir += '/%s' % scope
    # ng_model = NameGeneration(config, scope=scope)
    ng_model = VAE(config, scope=scope)
    return ng_model


def main(_):
    print(flags.FLAGS.__flags, '\n')

    dataset = get_name_data(FLAGS)
    ng_model = create_model(FLAGS)
    if FLAGS.is_train:
        # train(ng_model, dataset, FLAGS)
        train_vae(ng_model, dataset, FLAGS)


if __name__ == '__main__':
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    tf.app.run()

