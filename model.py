import tensorflow as tf
import os

from ops import *


class GAN(object):
    def __init__(self, sess,
                 d_input_dim, d_output_dim, max_time_step, min_grad, max_grad,
                 cell_dim, cell_layer_num, cell_keep_prob, char_dim,
                 hidden_dim, output_dr,
                 scope="NameGeneration", lr=1e-3):

        # session settings
        self.sess = sess
        self.scope = scope

        # hyper parameters
        self.lr = lr
        self.min_grad = min_grad
        self.max_grad = max_grad

        # model parameters
        self.max_time_step = max_time_step
        self.cell_dim = cell_dim
        self.cell_layer_num = cell_layer_num
        self.cell_keep_prob = cell_keep_prob
        self.char_dim = char_dim
        self.hidden_dim = hidden_dim
        self.output_dr = output_dr

        # input data placeholders
        self.d_input_dim = d_input_dim
        self.d_output_dim = d_output_dim
        self.inputs = tf.placeholder(tf.float32, [None, self.max_time_step, self.input_dim])
        self.input_len = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None, self.max_time_step])

        # model outputs
        self.logits = None
        self.losses = None
        self.grads = None
        self.mask = None

        # model settings
        self.optimizer = tf.train.AdamOptimizer()
        self.optimize = None
        self.params = None
        self.saver = None
        self.global_step = tf.Variable(0, name="step", trainable=False)

        # model build
        self.merged_summary = None
        self.train_writer = None
        self.valid_writer = None
        self.test_writer = None
        self.embed_writer = None
        self.build_model()

    def build_model(self):
        print("## Building an GAN model")
        fw_cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
        bw_cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)

        inputs_embed, projector, embed_config  = embedding_lookup(tf.argmax(self.inputs, 2), 
                self.input_dim, self.embed_dim, 'checkpoint/%s' % self.scope, scope='Character')
        inputs_reshape = rnn_reshape(inputs_embed, self.embed_dim, self.max_time_step)

        hidden1 = linear(inputs=rnn_model(inputs_reshape, self.input_len, fw_cell), 
                input_dim=self.cell_dim, 
                output_dim=self.hidden_dim,
                dropout_rate=self.output_dr,
                activation=tf.nn.relu, scope='Hidden1')
        '''
        hidden2 = linear(inputs=hidden1,
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                dropout_rate=self.output_dr, scope='Hidden2')
        '''

        self.logits = linear(inputs=hidden1,
            input_dim=self.hidden_dim,
            output_dim=self.output_dim, scope='Output')
 
        # Make grid of 0s and make weights [0 0 k 0 0] => [0 0 1 0 0]
        self.mask = tf.one_hot(self.input_len - 1, self.max_time_step)
        self.losses = tf.nn.seq2seq.sequence_loss(
            logits=[self.logits],
            targets=[tf.reshape(self.labels, [-1])],
            weights=[tf.reshape(self.mask, [-1])],
            average_across_timesteps=True,
            average_across_batch=True)

        tf.scalar_summary('Loss', self.losses)
        self.params = tf.trainable_variables()

        grads = []
        for grad in tf.gradients(self.losses, self.params):
            if grad is not None:
                grads.append(tf.clip_by_value(grad, self.min_grad, self.max_grad))
            else:
                grads.append(grad)

        self.grads = grads
        self.optimize = self.optimizer.apply_gradients(zip(grads, self.params), global_step=self.global_step)

        model_vars = [v for v in tf.global_variables()]
        print('model variables', [model_var.name for model_var in tf.global_variables()])
        self.saver = tf.train.Saver(model_vars)
        self.merged_summary = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter('./tensorboard/%s/train' % self.scope, self.sess.graph)
        self.valid_writer = tf.train.SummaryWriter('./tensorboard/%s/valid' % self.scope, self.sess.graph)
        self.test_writer = tf.train.SummaryWriter('./tensorboard/%s/test' % self.scope, self.sess.graph)
        self.embed_writer = tf.train.SummaryWriter('./checkpoint/%s' % self.scope)
        projector.visualize_embeddings(self.embed_writer, embed_config)

    def save(self, checkpoint_dir, step):
        file_name = "%s.model" % self.scope
        self.saver.save(self.sess, os.path.join(checkpoint_dir, file_name), global_step=step.astype(int))
        print("Model saved", file_name)

    def load(self, checkpoint_dir):
        file_name = "%s.model" % self.scope
        file_name += "-10800"
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, file_name))
        print("Model loaded", file_name)i

