import tensorflow as tf
import os

from tensorflow.contrib.tensorboard.plugins import projector


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def lstm_cell(cell_dim, layer_num, keep_prob):
    with tf.variable_scope('LSTM_Cell') as scope:
        cell = tf.nn.rnn_cell.BasicLSTMCell(cell_dim, forget_bias=1.0, activation=tf.tanh, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return tf.nn.rnn_cell.MultiRNNCell([cell] * layer_num, state_is_tuple=True)


def rnn_reshape(inputs, input_dim, max_time_step):
    with tf.variable_scope('Reshape') as scope:
        """
        reshape inputs from [batch_size, max_time_step, input_dim] to [max_time_step * (batch_size, input_dim)]

        :param inputs: inputs of shape [batch_size, max_time_step, input_dim]
        :param input_dim: dimension of input
        :param max_time_step: max of time step

        :return:
            outputs of shape [max_time_step * (batch_size, input_dim)]
        """
        inputs_tr = tf.transpose(inputs, [1, 0, 2])
        inputs_tr_reshape = tf.reshape(inputs_tr, [-1, input_dim])
        inputs_tr_reshape_split = tf.split(0, max_time_step, inputs_tr_reshape)
        return inputs_tr_reshape_split


def rnn_model(inputs, input_len, cell):
    with tf.variable_scope('RNN') as scope:
        outputs, state = tf.nn.rnn(cell, inputs, sequence_length=input_len, dtype=tf.float32, scope=scope)
        outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
        return outputs


def bi_rnn_model(inputs, input_len, fw_cell, bw_cell):
    with tf.variable_scope('Bi-RNN') as scope:
        outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs,
                sequence_length=input_len, dtype=tf.float32, scope=scope)
        outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
        return outputs


def embedding_lookup(inputs, voca_size, embedding_dim, visual_dir, scope='Embedding'):
    with tf.variable_scope(scope) as scope:
        embedding_table = tf.get_variable("embed", [voca_size, embedding_dim],
                initializer=tf.random_uniform_initializer(-1, 1), dtype=tf.float32)
        inputs_embed = tf.nn.embedding_lookup(embedding_table, inputs)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_table.name
        embedding.metadata_path = os.path.join(visual_dir, 'metadata.tsv')

        return inputs_embed, projector, config


def linear(inputs, input_dim, output_dim, dropout_rate=1.0, regularize_rate=0, activation=None, scope='Linear'):
    with tf.variable_scope(scope) as scope:
        inputs = tf.reshape(inputs, [-1, input_dim])
        weights = tf.get_variable('Weights', [input_dim, output_dim],
                                  initializer=tf.random_normal_initializer())
        variable_summaries(weights, scope.name + '/Weights')
        biases = tf.get_variable('Biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, scope.name + '/Biases')
        if activation is None:
            return dropout((tf.matmul(inputs, weights) + biases), dropout_rate)
        else:
            return dropout(activation(tf.matmul(inputs, weights) + biases), dropout_rate)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

