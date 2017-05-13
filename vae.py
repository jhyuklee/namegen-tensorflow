import tensorflow as tf
import numpy as np
import os

from ops import *
from model import NameGeneration


class VAE(NameGeneration):
    def __init__(self, config, scope="VAE"):
        self.latent_dim = config.latent_dim
        super(VAE, self).__init__(config, scope)
    
    def encoder(self, inputs, labels, input_len, reuse=False):
        """
        Args:
            inputs: inputs to encode with size [batch_size, time_steps

        Returns:
            z_mean: mean of latent space
            z_log_sigma_sq: squared log sigma of latent space
        """

        with tf.variable_scope('encoder', reuse=reuse):
            cell = lstm_cell(self.cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs_embed, projector, self.embed_config  = embedding_lookup(
                    inputs=inputs,
                    voca_size=self.input_dim,
                    embedding_dim=self.char_dim,
                    visual_dir='checkpoint/%s' % self.scope,
                    scope='Character')
            inputs_reshape = rnn_reshape(inputs_embed, self.char_dim, self.max_time_step)
            outputs, state = rnn_model(inputs_reshape, self.input_len, cell)
            outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
            indices = tf.concat(
                    axis=1,
                    values=[tf.expand_dims(tf.range(0, tf.shape(input_len)[0]), 1),
                            tf.expand_dims(input_len - 1, 1)]
                    )
            gathered_outputs = tf.gather_nd(outputs, indices)
            print('gathered_outputs', gathered_outputs)
            concat_outputs = tf.concat(
                    [tf.one_hot(labels, self.class_dim), gathered_outputs], 1)
            print('concat_output', concat_outputs)
            z_mean = linear(inputs=concat_outputs,
                    output_dim=self.latent_dim, 
                    scope='z_mean')
            z_log_sigma_sq = linear(inputs=concat_outputs,
                    output_dim=self.latent_dim, 
                    scope='z_sigma')
            print('mean and sigma', z_mean, z_log_sigma_sq)

            return z_mean, z_log_sigma_sq

    def decoder(self, inputs, state, feed_prev=False, reuse=None):
        """
        Args:
            z: latent variable created by encoder

        Returns:
            decoded: decoded outputs with size [batch_size, time_steps, input_dim]
        """

        with tf.variable_scope('decoder', reuse=reuse):
            # make dummy linear for loop function
            dummy = linear(inputs=tf.constant(1, tf.float32, [100, self.latent_dim]),
                    output_dim=self.input_dim, 
                    scope='rnn_decoder/loop_function/Out', reuse=reuse)

            if feed_prev:
                def loop_function(prev, i):
                    next = tf.argmax(linear(inputs=prev,
                        output_dim=self.input_dim,
                        scope='Out', reuse=True), 1)
                    return tf.one_hot(next, self.input_dim)
            else:
                loop_function = None

            cell = lstm_cell(self.latent_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs = tf.one_hot(inputs, self.input_dim)
            inputs_t = tf.unstack(tf.transpose(inputs, [1, 0, 2]), self.max_time_step)
            outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(inputs_t, state, cell, loop_function)
            outputs_t = tf.transpose(tf.stack(outputs), [1, 0, 2])
            outputs_tr = tf.reshape(outputs_t, [-1, self.latent_dim])
            decoded = linear(inputs=outputs_tr,
                    output_dim=self.input_dim, scope='rnn_decoder/loop_function/Out', reuse=True)
            return decoded

    def build_model(self):
        z_mean, z_log_sigma_sq = self.encoder(self.inputs, self.labels, self.input_len)
        eps = tf.random_normal([tf.shape(self.inputs)[0], self.latent_dim], 0, 1)
        self.z = tf.add(z_mean, tf.sqrt(tf.exp(z_log_sigma_sq)) * eps)
        self.decoded = self.decoder(self.decoder_inputs, ((tf.zeros_like(self.z), self.z),))

        reconstr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.decoded, labels=tf.reshape(self.inputs, [-1]))
        latent_loss = 0.5 * tf.reduce_sum(-1 - z_log_sigma_sq + tf.square(z_mean)
                + tf.exp(z_log_sigma_sq), 1)
        self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss)
         
