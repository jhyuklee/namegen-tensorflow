import tensorflow as tf
import numpy as np
import time
import sys
import os
import operator
import random
import re

from random import shuffle
from utils import *


def one_hot(index, length, value=1):
    assert index >= 0 and index < length, 'index must be bigger or equal than 0'
    vector = np.zeros([length])
    vector[index] = value
    return vector


def get_name_data(config):
    for root, dir, files in os.walk(config.data_dir):
        inputs = []
        inputs_length = []
        decoder_inputs = []
        labels = []
        
        char2idx = {}
        idx2char = {}
        country2idx = {}
        idx2country = {}
        name_dict = {}
        max_len = 0
        vocab_size = 0
        PAD, GO, EOS = 0, 0, 0

        max_name_len = 45
        class_dim = config.class_dim

        for file_cnt, file_name in enumerate(sorted(files)):
            data = open(os.path.join(root, file_name))
            file_len = 0
            
            if file_name == 'char_to_idx.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    char, index = line[:-1].split('\t')
                    char2idx[char] = int(index)
                    idx2char[int(index)] = char
                    
                # Add PAD, GO, EOS
                char2idx['PAD'] = len(char2idx) 
                char2idx['GO'] = len(char2idx) 
                char2idx['EOS'] = len(char2idx) 
                vocab_size = len(char2idx)
                idx2char[vocab_size-3] = 'PAD' 
                idx2char[vocab_size-2] = 'GO'
                idx2char[vocab_size-1] = 'EOS' 

            elif file_name == 'country_to_idx.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    country, index = line[:-1].split('\t')
                    country2idx[country] = int(index)
                    idx2country[int(index)] = country

            elif file_name == 'name_to_country.txt':
                PAD = vocab_size - 3
                GO = vocab_size - 2
                EOS = vocab_size - 1
                for k, line in enumerate(data):
                    raw_name, nationality = line[:-1].split('\t')
                    raw_name = re.sub(r'\ufeff', '', raw_name)    # delete BOM
                    
                    name = [one_hot(char2idx[c], vocab_size) for c in raw_name]
                    decoder_name = np.insert(name[:], 0, one_hot(GO, vocab_size), axis=0)
                    decoder_name = np.append(decoder_name[:], [one_hot(EOS, vocab_size)], axis=0)
                    nationality = one_hot(country2idx[nationality], class_dim)
                    name_length = len(name)

                    if max_len < len(name): # update the maximum length
                        max_len = len(name)
                    while len(name) != max_name_len: # fill with PAD
                        name.append(one_hot(PAD, vocab_size))
                    while len(decoder_name) != max_name_len:
                        decoder_name = np.append(decoder_name[:], [one_hot(PAD, vocab_size)], axis=0)

                    name_s = ''.join([idx2char[char] for char in np.argmax(name, 1)][:name_length])
                    name_dict[name_s] = 0

                    inputs.append(name)
                    decoder_inputs.append(decoder_name)
                    labels.append(nationality)
                    inputs_length.append(name_length)
                    file_len = k + 1
            else:
                pass 

            if file_len > 0:
                print('reading', file_name, 'of length', file_len)

    # shuffle
    pairs = list(zip(inputs, inputs_length, labels, decoder_inputs))
    shuffle(pairs)
    inputs, inputs_length, labels, decoder_inputs = zip(*pairs)

    # To np-array
    inputs = np.array(inputs)
    inputs_length = np.array(inputs_length)
    labels = np.array(labels)
    decoder_inputs = np.array(decoder_inputs)
    
    print('\n## Data stats')
    print('vocab size: %d' % vocab_size)
    print('name max length:', max_len, '/', max_name_len)
    print('unique name set:', len(name_dict))
    name_sorted = sorted(name_dict.items(), key=operator.itemgetter(1))
    print(name_sorted[::-1][:10])
    print('data shapes:', inputs.shape, decoder_inputs.shape, labels.shape, inputs_length.shape)
    name_s = ''.join([idx2char[char] for char in np.argmax(inputs[0], 1)][:inputs_length[0]])

    print('\n## Data sample')
    print(np.argmax(inputs[0], 1))
    print(np.argmax(decoder_inputs[0], 1))
    print('name:', name_s)
    print('label:', idx2country[np.argmax(labels[0], 0)] + ', length:', inputs_length[0], '\n')

    return (inputs, decoder_inputs, labels, inputs_length, 
            [idx2char, char2idx], [idx2country, country2idx])


def train(model, dataset, config):
    sess = model.session
    batch_size = config.batch_size
    inputs, decoder_inputs, labels, inputs_length, char_set, country_set = dataset
    idx2char, char2idx = char_set
    idx2country, country2idx = country_set

    print('\n## Autoencoder Training')
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if config.load_autoencoder is True:
        model.load('checkpoint/pretrained_ae', file_name=config.pretrained_ae)
    
    for epoch_idx in range(config.ae_epoch if config.train_autoencoder else 1):
        for datum_idx in range(0, len(inputs), batch_size):
            batch_inputs = inputs[datum_idx:datum_idx+batch_size]
            batch_decoder_inputs = decoder_inputs[datum_idx:datum_idx+batch_size]
            batch_input_len = inputs_length[datum_idx:datum_idx + batch_size]
            batch_labels = labels[datum_idx:datum_idx+batch_size]
            batch_inputs_noise = np.random.normal(0, 1, (len(batch_inputs), config.char_dim))
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_inputs_noise) == len(batch_decoder_inputs), 'not the same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.inputs_noise: batch_inputs_noise, model.labels: batch_labels, 
                    model.decoder_inputs: batch_decoder_inputs}
            
            if config.train_autoencoder:
                sess.run(model.ae_optimize, feed_dict=feed_dict)

            if (datum_idx % (batch_size*5) == 0) or (datum_idx + batch_size >= len(inputs)):
                decoded, ae_loss = sess.run([model.decoded, model.ae_loss], feed_dict=feed_dict)
                decoded = decoded.reshape((len(batch_inputs), config.max_time_step, config.input_dim))
                decoded_name = ''.join([idx2char[char] 
                    for char in np.argmax(decoded[0], 1)])[:batch_input_len[0]]
                original_name = ''.join([idx2char[char] 
                    for char in np.argmax(batch_inputs[0], 1)])[:batch_input_len[0]]
                _progress = "\rEp %d: %s/%s, ae_loss: %.3f" % (epoch_idx, original_name,
                        decoded_name, ae_loss)
                sys.stdout.write(_progress)
                sys.stdout.flush()
        print()

    if config.train_autoencoder is True:
        model.save(config.checkpoint_dir, file_name=config.pretrained_ae)


    print('\n## GAN Training')
    d_iter = 1
    g_iter = 1
    for epoch_idx in range(config.gan_epoch):
        for datum_idx in range(0, len(inputs), batch_size):
            batch_inputs = inputs[datum_idx:datum_idx+batch_size]
            batch_decoder_inputs = decoder_inputs[datum_idx:datum_idx+batch_size]
            batch_input_len = inputs_length[datum_idx:datum_idx + batch_size]
            batch_labels = labels[datum_idx:datum_idx+batch_size]

            batch_z = np.random.normal(0, 1, (len(batch_inputs), config.input_dim))
            batch_inputs_noise = np.random.normal(0, 1, (len(batch_inputs), config.char_dim))
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_z) == len(batch_decoder_inputs), 'not the same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.z: batch_z, model.labels: batch_labels, model.decoder_inputs:
                    batch_decoder_inputs, model.inputs_noise: batch_inputs_noise}

            for _ in range(d_iter):
                sess.run(model.d_optimize, feed_dict=feed_dict)
            for _ in range(g_iter):
                sess.run(model.g_optimize, feed_dict=feed_dict)

            if (datum_idx % (batch_size*5) == 0) or (datum_idx + batch_size >= len(inputs)):
                d_loss, g_loss, g_decoded = sess.run([model.d_loss, model.g_loss, model.g_decoded], 
                        feed_dict=feed_dict)
                g_decoded = g_decoded.reshape((len(batch_inputs), config.max_time_step, 
                    config.input_dim))
                g_decoded_name = ''.join([idx2char[char] for char in np.argmax(g_decoded[0], 1)])
                if char2idx['PAD'] in np.argmax(g_decoded[0], 1):
                    PAD_idx = np.argwhere(np.argmax(g_decoded[0], 1) == char2idx['PAD'])
                    PAD_idx = PAD_idx.flatten().tolist()[0]
                else:
                    PAD_idx = -1
                # _progress = progress((datum_idx + batch_size) / float(len(inputs)))
                _progress = "\rEp %d d:%.3f, g:%.3f, %s (%s)" % \
                        (epoch_idx, d_loss, g_loss, g_decoded_name[:PAD_idx], 
                                idx2country[np.argmax(batch_labels[0], 0)])
                sys.stdout.write(_progress)
                sys.stdout.flush()

                f = open(config.results_dir + '/' + model.scope, 'w')
                for decoded, label in zip(g_decoded, batch_labels):
                    name = ''.join([idx2char[char] for char in np.argmax(decoded, 1)])
                    if char2idx['PAD'] in np.argmax(decoded, 1):
                        PAD_idx = np.argwhere(np.argmax(decoded, 1) == char2idx['PAD'])
                        PAD_idx = PAD_idx.flatten().tolist()[0]
                    else:
                        PAD_idx = -1
                    f.write(name[:PAD_idx] + '\t' + idx2country[np.argmax(label, 0)] + '\n')
                f.close()
     
        print()

    model.save(config.checkpoint_dir, sess.run(model.global_step))

