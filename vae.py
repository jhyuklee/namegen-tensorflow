import tensorflow as tf
import numpy as np
import os

from ops import *
from model import NameGeneration


class VAE(NameGeneration):
    def __init__(self, config, scope="VAE"):
        self.latent_dim = config.latent_dim
        self.vae_lr = config.vae_lr
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
            indices = tf.concat(
                    axis=1,
                    values=[tf.expand_dims(tf.range(0, tf.shape(input_len)[0]), 1),
                            tf.expand_dims(input_len - 1, 1)]
                    )
            gathered_outputs = tf.gather_nd(outputs, indices)
            concat_outputs = tf.concat(
                    [tf.one_hot(labels, self.class_dim), gathered_outputs], 1)
            z_mean = linear(inputs=concat_outputs,
                    output_dim=self.latent_dim, 
                    scope='z_mean')
            z_log_sigma_sq = linear(inputs=concat_outputs,
                    output_dim=self.latent_dim, 
                    scope='z_sigma')

            return z_mean, z_log_sigma_sq

    def decoder(self, inputs, state, feed_prev=False, reuse=None):
        """
        Args:
            z: latent variable created by encoder

        Returns:
            decoded: decoded outputs with size [batch_size, time_steps, input_dim]
        """

        with tf.variable_scope('decoder', reuse=reuse):
            cell_dim = self.latent_dim + self.class_dim
            # make dummy linear for loop function
            dummy = linear(inputs=tf.constant(1, tf.float32, [100, cell_dim]),
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

            cell = lstm_cell(cell_dim, self.cell_layer_num, self.cell_keep_prob)
            inputs = tf.one_hot(inputs, self.input_dim)
            inputs_t = tf.unstack(tf.transpose(inputs, [1, 0, 2]), self.max_time_step)
            outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(inputs_t, state, cell, loop_function)
            outputs_t = tf.transpose(tf.stack(outputs), [1, 0, 2])
            outputs_tr = tf.reshape(outputs_t, [-1, cell_dim])
            decoded = linear(inputs=outputs_tr,
                    output_dim=self.input_dim, scope='rnn_decoder/loop_function/Out', reuse=True)
            return decoded

    def build_model(self):
        z_mean, z_log_sigma_sq = self.encoder(self.inputs, self.labels, self.input_len)
        eps = tf.random_normal([tf.shape(self.inputs)[0], self.latent_dim], 0, 1)
        self.z = tf.add(z_mean, tf.sqrt(tf.exp(z_log_sigma_sq)) * eps)
        y_z = tf.concat(axis=1, values=[self.z, tf.one_hot(self.labels, self.class_dim)])
        self.decoded = self.decoder(self.decoder_inputs, ((tf.zeros_like(y_z), y_z),))

        reconstr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.decoded, labels=tf.reshape(self.inputs, [-1])))
        latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(-1 - z_log_sigma_sq + tf.square(z_mean)
                + tf.exp(z_log_sigma_sq), 1))
        self.vae_loss = reconstr_loss + latent_loss
        
        self.params = tf.trainable_variables()
        vae_vars = [var for var in self.params if 'encoder' in var.name or 'decoder' in var.name]
        self.vae_optimize = tf.train.AdamOptimizer(self.vae_lr).minimize(self.vae_loss, var_list=vae_vars)
        model_vars = [v for v in tf.global_variables()]
        self.saver = tf.train.Saver(model_vars)

    def sample(self, c, num_sample, feed_dict):
        z = tf.random_normal([num_sample, self.latent_dim], 0, 1)
        y = tf.tile(tf.expand_dims(tf.one_hot(c, self.class_dim), 0), [num_sample, 1])
        y_z = tf.concat(axis=1, values=[z, y])
        decoded = self.decoder(self.decoder_inputs, ((tf.zeros_like(y_z), y_z),), reuse=True)
        decoded = tf.argmax(tf.reshape(decoded, [num_sample, self.max_time_step, self.input_dim]), 2)

        return self.session.run(decoded, feed_dict=feed_dict)
        

