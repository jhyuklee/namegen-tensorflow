import tensorflow as tf
import numpy as np
import os

from ops import *


class GAN(object):
    def __init__(self, config, scope="NameGeneration"):

        # session settings
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_config)
        self.scope = scope

        # hyper parameters
        self.ae_lr = config.ae_lr
        self.cf_lr = config.cf_lr
        self.gan_lr = config.gan_lr
        self.min_grad = config.min_grad
        self.max_grad = config.max_grad

        # model parameters
        self.max_time_step = config.max_time_step
        self.cell_dim = config.cell_dim
        self.cell_layer_num = config.cell_layer_num
        self.cell_keep_prob = config.cell_keep_prob
        self.char_dim = config.char_dim
        self.hidden_dim = config.hidden_dim
        self.output_dr = config.output_dr

        # input data placeholders
        self.input_dim = config.input_dim
        self.class_dim = config.class_dim
        self.inputs = tf.placeholder(tf.int64, [None, self.max_time_step])
        self.inputs_noise = tf.placeholder(tf.float32, [None, self.char_dim])
        self.input_len = tf.placeholder(tf.int32, [None])
        self.decoder_inputs = tf.placeholder(tf.int64, [None, self.max_time_step])
        self.z = tf.placeholder(tf.float32, [None, self.input_dim])
        self.labels = tf.placeholder(tf.int64, [None])
        self.global_step = tf.Variable(0, name="step", trainable=False)

        # model build
        self.build_model()
        self.session.run(tf.global_variables_initializer())

    def generator(self, zc, reuse=False):
        """
        Args:
            zc: random input vector of size [batch_size, z_dim + class_dim]

        Returns:
            out: generated name hidden vectors of size [batch_size, cell_dim]
        """

        with tf.variable_scope('generator', reuse=reuse):
            hidden1 = linear(inputs=zc,
                    output_dim=self.hidden_dim * 2,
                    activation=tf.nn.relu, scope='Hidden1')
            out = linear(inputs=hidden1,
                    output_dim=self.cell_dim, scope='Out')

            return out

    def discriminator(self, inputs, reuse=False):
        """
        Args:
            inputs: inputs of size [batch_size, cell_dim*2 + class_dim]

        Returns:
            logits: unnormalized output probabilities of size [batch_size]
        """

        with tf.variable_scope('discriminator', reuse=reuse):
            hidden1 = linear(inputs=inputs,
                    output_dim=self.hidden_dim,
                    activation=tf.nn.relu, scope='Hidden1')
            hidden2 = linear(inputs=hidden1,
                    output_dim=self.hidden_dim,
                    activation=tf.nn.relu, scope='Hidden2')
            logits = linear(inputs=hidden1,
                    output_dim=1, scope='Out')

            return logits

    def encoder(self, inputs, inputs_noise=None, reuse=False):
        """
        Args:
            inputs: inputs to encode with size [batch_size, time_steps, input_dim]

        Returns:
            state: hidden state vector of encoder with size [batch_size, cell_dim*2]
        """

        with tf.variable_scope('encoder', reuse=reuse):
            cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs_embed, projector, self.embed_config  = embedding_lookup(
                    inputs=inputs,
                    voca_size=self.input_dim,
                    embedding_dim=self.char_dim,
                    visual_dir='checkpoint/%s' % self.scope,
                    scope='Character')
            if inputs_noise is not None:
                inputs_noise = tf.expand_dims(inputs_noise, 1)
                inputs_embed = tf.add(inputs_embed, inputs_noise)
            inputs_reshape = rnn_reshape(inputs_embed, self.char_dim, self.max_time_step)
            outputs, state = rnn_model(inputs_reshape, self.input_len, cell)
            return state

    def decoder(self, inputs, state, feed_prev=False, reuse=None):
        """
        Args:
            inputs: decoder inputs with size [batch_size, time_steps, input_dim]
            state: hidden state of encoder with size [batch_size, cell_dim]

        Returns:
            decoded: decoded outputs with size [batch_size, time_steps, input_dim]
        """

        with tf.variable_scope('decoder', reuse=reuse):
            # make dummy linear for loop function
            dummy = linear(inputs=tf.constant(1, tf.float32, [100, self.cell_dim]),
                    output_dim=self.input_dim, scope='rnn_decoder/loop_function/Out', reuse=reuse)

            if feed_prev:
                def loop_function(prev, i):
                    next = tf.argmax(linear(inputs=prev,
                        output_dim=self.input_dim,
                        scope='Out', reuse=True), 1)
                    return tf.one_hot(next, self.input_dim)
            else:
                loop_function = None

            cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs = tf.one_hot(inputs, self.input_dim)
            inputs_t = tf.unstack(tf.transpose(inputs, [1, 0, 2]), self.max_time_step)
            outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(inputs_t, state, cell, loop_function)
            outputs_t = tf.transpose(tf.stack(outputs), [1, 0, 2])
            outputs_tr = tf.reshape(outputs_t, [-1, self.cell_dim])
            decoded = linear(inputs=outputs_tr,
                    output_dim=self.input_dim, scope='rnn_decoder/loop_function/Out', reuse=True)
            return decoded

    def classifier(self, state, reuse=None):
        """
        Args:
            state: final state from decoder to classify

        Returns:
            logits: unnormalized probability distribution for class labels
        """
        with tf.variable_scope('classifier', reuse=reuse):
            hidden = linear(inputs=state,
                    output_dim=self.hidden_dim,
                    activation=tf.nn.relu, scope='Hidden1')
            logits = linear(inputs=hidden,
                    output_dim=self.class_dim, scope='Out')

            return logits



    def build_model(self):
        # encoder decoder loss
        # state = self.encoder(self.inputs, self.inputs_noise)
        state = self.encoder(self.inputs)
        self.decoded = self.decoder(self.decoder_inputs,
                ((tf.zeros_like(state[0][1]), state[0][1]),), feed_prev=True)
        self.ae_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.decoded, labels=tf.reshape(self.inputs, [-1])))

        # classifier loss
        cf_logits = self.classifier(state[0][1])
        self.cf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=cf_logits, labels=self.labels))
        self.cf_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(cf_logits, 1), self.labels), tf.float32))

        # generator logits
        h_hat = self.generator(tf.concat([self.z, tf.one_hot(self.labels, self.class_dim)], 1))
        logits_fake = self.discriminator(tf.concat([h_hat, tf.one_hot(self.labels, self.class_dim)], 1))
        # logits_fake = self.discriminator(h_hat)
        cf_logits_fake = self.classifier(h_hat, reuse=True)
        self.g_decoded = self.decoder(self.decoder_inputs, ((tf.zeros_like(h_hat), h_hat),),
                feed_prev=True, reuse=True)

        # discriminator logits
        h = self.encoder(self.inputs, reuse=True)
        logits_real = self.discriminator(tf.concat([h[0][1], tf.one_hot(self.labels, self.class_dim)], 1), reuse=True)
        # logits_real = self.discriminator(h[0][1], reuse=True)

        # compute loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,
            labels=tf.ones_like(logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
            labels=tf.zeros_like(logits_fake)))
        self.cf_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=cf_logits_fake, labels=self.labels))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
            labels=tf.ones_like(logits_fake)))
        # self.g_loss = (self.g_loss + self.cf_loss_fake) / 2

        tf.summary.scalar('Discriminator Loss', self.d_loss)
        tf.summary.scalar('Generator Loss', self.g_loss)

        self.params = tf.trainable_variables()
        d_vars = [var for var in self.params if 'discriminator' in var.name]
        g_vars = [var for var in self.params if 'generator' in var.name]
        ae_vars = [var for var in self.params if 'encoder' in var.name or 'decoder' in var.name]
        cf_vars = [var for var in self.params if 'classifier' in var.name]
        print('autoencoder variables', [model_var.name for model_var in ae_vars])
        print('classifier variables', [model_var.name for model_var in cf_vars])
        print('discriminator variables', [model_var.name for model_var in d_vars])
        print('generator variables', [model_var.name for model_var in g_vars])

        self.d_optimize = tf.train.AdamOptimizer(self.gan_lr).minimize(self.d_loss, var_list=d_vars)
        self.g_optimize = tf.train.AdamOptimizer(self.gan_lr).minimize(self.g_loss, var_list=g_vars)
        self.ae_optimize = tf.train.AdamOptimizer(self.ae_lr).minimize(self.ae_loss, var_list=ae_vars)
        self.cf_optimize = tf.train.AdamOptimizer(self.cf_lr).minimize(self.cf_loss, var_list=cf_vars)

        model_vars = [v for v in tf.global_variables()]
        self.saver = tf.train.Saver(model_vars)

    def save(self, model_path):
        file_name = "%s.model" % self.scope
        self.saver.save(self.session, os.path.join(model_path, file_name))
        print("Model saved", os.path.join(model_path, file_name))

    def load(self, model_path):
        self.saver.restore(self.session, model_path)
        print("Model loaded", model_path)

