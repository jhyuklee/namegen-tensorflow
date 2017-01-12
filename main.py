import tensorflow as tf

from time import gmtime, strftime
from dataset import *
from model import GAN


flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train")
flags.DEFINE_integer("input_dim", 48, "Data input dimension")
flags.DEFINE_integer("max_time_step", 50, "Maximum time step of RNN")
flags.DEFINE_integer("min_grad", -10, "Minimum gradient to clip")
flags.DEFINE_integer("max_grad", 10, "Maximum gradient to clip")
flags.DEFINE_integer("cell_dim", 200, "Dimension of RNN cell")
flags.DEFINE_integer("cell_layer_num", 1, "The layer number of RNN ")
flags.DEFINE_integer("cell_keep_prob", 1.0, "Keep prob of RNN cell dropout")
flags.DEFINE_integer("char_dim", 50, "Dimension of character embedding")
flags.DEFINE_integer("hidden_dim", 300, "Dimension of hidden layer for FFNN")
flags.DEFINE_integer("output_dr", 0.5, "Dropout rate of FFNN")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing")
flags.DEFINE_boolean("continue_train", False, "True to continue training from saved checkpoint. False for restarting.")
FLAGS = flags.FLAGS


def create_model(config, sess):
    scope = 'NameGeneration-' + strftime("%Y%m%d%H%M%S", gmtime())
    config.checkpoint_dir += '/%s' % scope
    print(scope)
    
    gan_model = GAN(sess=sess,
                    input_dim=config.input_dim,
                    max_time_step=config.max_time_step,
                    min_grad=config.min_grad, max_grad=config.max_grad,
                    cell_dim=config.cell_dim,
                    cell_layer_num=config.cell_layer_num,
                    cell_keep_prob=config.cell_keep_prob,
                    char_dim=config.char_dim,
                    hidden_dim=config.hidden_dim,
                    output_dr=config.output_dr,
                    scope=scope)
    return gan_model


def main(_):
    config = tf.ConfigProto(
            device_count={'GPU':1}
    )
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        gan_model = create_model(FLAGS, sess)
        print(flags.FLAGS.__flags, '\n')

        if FLAGS.is_train:
            train(gan_model, FLAGS, sess)
        else:
            test_cost, test_acc = test(gan_model, FLAGS, sess, is_valid=False)
            print("Testing loss: %.3f, accuracy: %.3f" % (test_cost, test_acc))


if __name__ == '__main__':
    tf.app.run()

