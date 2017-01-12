import tensorflow as tf
import os

from ops import *


class GAN(object):
    def __init__(self, sess,
                 input_dim, max_time_step, min_grad, max_grad,
                 cell_dim, cell_layer_num, cell_keep_prob, char_dim,
                 hidden_dim, output_dr,
                 scope="NameGeneration", lr=1e-2):

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
        self.input_dim = input_dim
        self.inputs = tf.placeholder(tf.float32, [None, self.max_time_step, self.input_dim])
        self.input_len = tf.placeholder(tf.int32, [None])
        self.z = tf.placeholder(tf.float32, [None, self.input_dim])
        self.labels = tf.placeholder(tf.int32, [None, self.max_time_step])   # future works: conditional gan

        # model outputs
        self.d_loss = None
        self.g_loss = None
        self.grads = None
        self.mask = None

        # model settings
        self.optimizer = tf.train.AdamOptimizer()
        self.d_optimize_real = None
        self.d_optimize_fake = None
        self.g_optimize = None
        self.params = None
        self.saver = None
        self.global_step = tf.Variable(0, name="step", trainable=False)

        # model build
        self.merged_summary = None
        self.train_writer = None
        self.valid_writer = None
        self.test_writer = None
        self.embed_writer = None
        self.embed_config = None
        self.build_model()

    def generator(self, z, reuse=False):
        """
        Args:
            z: random input vector of size [batch_size, z_dim]

        Returns:
            out: generated name hidden vectors of size [batch_size, cell_dim]
        """

        with tf.variable_scope('generator', reuse=reuse):
            hidden1 = linear(inputs=z,
                    output_dim=self.hidden_dim * 2,
                    activation=tf.nn.relu, scope='Hidden1')
            out = linear(inputs=hidden1,
                    output_dim=self.cell_dim, scope='Out')

            return out

    def discriminator(self, inputs, reuse=False):
        """
        Args:
            inputs: inputs of size [batch_size, cell_dim]

        Returns:
            logits: unnormalized output probabilities of size [batch_size]
        """

        with tf.variable_scope('discriminator', reuse=reuse):
            hidden1 = linear(inputs=inputs,
                    output_dim=self.hidden_dim,
                    activation=tf.nn.relu, scope='Hidden1')
            logits = linear(inputs=hidden1,
                    output_dim=1, scope='Out')

            return logits

    def encoder(self, inputs, reuse=False):
        """
        Args:
            inputs: inputs to encode with size [batch_size, time_steps, input_dim]

        Returns:
            state: hidden state vector of encoder with size [batch_size, cell_dim]
        """

        with tf.variable_scope('encoder', reuse=reuse):
            cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs_embed, projector, self.embed_config  = embedding_lookup(tf.argmax(inputs, 2), 
                    self.input_dim, self.char_dim, 'checkpoint/%s' % self.scope, scope='Character')
            inputs_reshape = rnn_reshape(inputs_embed, self.char_dim, self.max_time_step)
            outputs, state = rnn_model(inputs_reshape, self.input_len, cell) 
            return state[0][1]

    def decoder(self, inputs, state, reuse=False):
        """
        Args:
            inputs: inputs to decode with size [batch_size, time_steps, input_dim]
            state: hidden state of encoder with size [batch_size, cell_dim]

        Returns:
            outputs_embed: decoded outputs with size [batch_size, char_dim]
        """

        with tf.variable_scope('decoder', reuse=reuse):
            cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs_t = tf.unpack(tf.transpose(inputs, [1, 0, 2]), self.max_time_step)
            # vocab[vocab_size-1] = GO, vocab[vocab_size-2] = EOS
            inputs_pad = [tf.one_hot(0, self.input_dim), inputs_t, tf.one_hot(1, self.input_dim)]
            print(inputs_pad)
            outputs, states = tf.nn.seq2seq.rnn_decoder(inputs_pad, state, cell)

        with tf.variable_scope('encoder', reuse=reuse):
            outputs_embed, projector, self.embed_config  = embedding_lookup(tf.argmax(outputs, 2), 
                    self.input_dim, self.char_dim, 'checkpoint/%s' % self.scope, scope='Character')
            
            return outputs_embed


    def build_model(self):
        print("## Building an GAN model")

        # generator loss
        h_hat = self.generator(self.z)
        logits_fake = self.discriminator(h_hat)
        
        # discriminator loss
        h = self.encoder(self.inputs)
        logits_real = self.discriminator(h, reuse=True)

        # compute loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_real,
            tf.ones_like(logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake,
            tf.zeros_like(logits_fake)))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake,
            tf.ones_like(logits_fake)))

        tf.summary.scalar('Discriminator Loss', self.d_loss)
        tf.summary.scalar('Generator Loss', self.g_loss)
        
        self.params = tf.trainable_variables()
        d_vars = [var for var in self.params if 'discriminator' in var.name]
        g_vars = [var for var in self.params if 'generator' in var.name]
        e_vars = [var for var in self.params if 'encoder' in var.name]

        self.d_optimize_real = self.optimizer.minimize(d_loss_real, var_list=d_vars)
        self.d_optimize_fake = self.optimizer.minimize(d_loss_fake, var_list=d_vars)
        self.g_optimize = self.optimizer.minimize(self.g_loss, var_list=g_vars)

        model_vars = [v for v in tf.global_variables()]
        print('model variables', [model_var.name for model_var in tf.global_variables()])
        self.saver = tf.train.Saver(model_vars)
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./tensorboard/%s/train' % self.scope, self.sess.graph)
        self.valid_writer = tf.summary.FileWriter('./tensorboard/%s/valid' % self.scope, self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./tensorboard/%s/test' % self.scope, self.sess.graph)
        self.embed_writer = tf.summary.FileWriter('./checkpoint/%s' % self.scope)
        projector.visualize_embeddings(self.embed_writer, self.embed_config)

    def save(self, checkpoint_dir, step):
        file_name = "%s.model" % self.scope
        self.saver.save(self.sess, os.path.join(checkpoint_dir, file_name), global_step=step.astype(int))
        print("Model saved", file_name)

    def load(self, checkpoint_dir, step=None):
        file_name = "%s.model" % self.scope
        file_name += ("-" + str(step))
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, file_name))
        print("Model loaded", file_name)

