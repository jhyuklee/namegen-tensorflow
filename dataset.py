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


def select_data(dataset, class_info, is_all=True):
    if not is_all:
        idx2country, country2cnt = class_info
        new_dataset = []
        for item in dataset:
            if idx2country[item[2]] in country2cnt:
                # TODO: add new onehot label
                new_dataset.append(item)
        return new_dataset
    else:
        return dataset


def get_name_data(config):
    for root, dir, files in os.walk(config.data_dir):
        inputs = []
        inputs_length = []
        decoder_inputs = []
        labels = []
        
        char2idx = {}
        idx2char = {}
        country2cnt = {}
        country2idx = {}
        idx2country = {}
        name_dict = {}
        max_len = 0
        vocab_size = 0
        PAD, GO, EOS = 0, 0, 0
        max_name_len = 45

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
                    country2cnt[country] = 0
                    idx2country[int(index)] = country

            elif file_name == 'new_names.txt':
                PAD = vocab_size - 3
                GO = vocab_size - 2
                EOS = vocab_size - 1
                for k, line in enumerate(data):
                    _progress = progress(k / 1787194) + "Reading %d names... "%(k)
                    sys.stdout.write(_progress)
                    sys.stdout.flush()
                    if k >= 200000: break

                    raw_name, nationality = line[:-1].split('\t')
                    raw_name = re.sub(r'\ufeff', '', raw_name)    # delete BOM
                    
                    name = [char2idx[c] for c in raw_name]
                    decoder_name = np.insert(name[:], 0, GO, axis=0)
                    decoder_name = np.append(decoder_name[:], [EOS], axis=0)
                    country2cnt[nationality] += 1
                    nationality = country2idx[nationality]
                    name_length = len(name)

                    if max_len < len(name): # update the maximum length
                        max_len = len(name)
                    while len(name) != max_name_len: # fill with PAD
                        name.append(PAD)
                    while len(decoder_name) != max_name_len:
                        decoder_name = np.append(decoder_name[:], [PAD], axis=0)

                    name_s = ''.join([idx2char[char] for char in name][:name_length])
                    name_dict[name_s] = 0

                    inputs.append(name)
                    decoder_inputs.append(decoder_name)
                    labels.append(nationality)
                    inputs_length.append(name_length)
                    file_len = k + 1
                print()
            else:
                pass 

            if file_len > 0:
                print('reading', file_name, 'of length', file_len)

    # shuffle
    pairs = list(zip(inputs, inputs_length, labels, decoder_inputs))
    shuffle(pairs)
    inputs, inputs_length, labels, decoder_inputs = zip(*pairs)

    print('\n## Data stats')
    print('vocab size: %d' % vocab_size)
    print('name max length:', max_len, '/', max_name_len)
    print('unique name set:', len(name_dict))
    name_sorted = sorted(name_dict.items(), key=operator.itemgetter(1))
    print(name_sorted[::-1][:10])
    country_sorted = sorted(country2cnt.items(), key=operator.itemgetter(1))[::-1][3:4]
    print(country_sorted)

    # Select only majority class items
    dataset = list(zip(inputs, decoder_inputs, labels, inputs_length))
    inputs, decoder_inputs, labels, inputs_length = \
             zip(*select_data(dataset, [idx2country, dict(country_sorted)], is_all=False))
    print('select from', len(dataset), 'to', len(inputs))
    
    # To np-array
    inputs = np.array(inputs)
    inputs_length = np.array(inputs_length)
    labels = np.array(labels)
    decoder_inputs = np.array(decoder_inputs)
    new_dataset = (inputs, decoder_inputs, labels, inputs_length)
    print('data shapes:', new_dataset[0].shape, new_dataset[1].shape, new_dataset[2].shape, 
            new_dataset[3].shape)

    print('\n## Data sample')
    print(inputs[0])
    print(decoder_inputs[0])
    name_s = ''.join([idx2char[char] for char in inputs[0]][:inputs_length[0]])
    print('name:', name_s)
    print('label:', idx2country[labels[0]] + ', length:', inputs_length[0], '\n')

    return (new_dataset, [idx2char, char2idx], [idx2country, country2idx])


def train(model, dataset, config):
    sess = model.session
    batch_size = config.batch_size
    sets, char_set, country_set = dataset
    inputs, decoder_inputs, labels, inputs_length = sets
    idx2char, char2idx = char_set
    idx2country, country2idx = country_set

    print('\n## Autoencoder Training')
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if config.load_autoencoder is True:
        model.load('checkpoint/' + config.pretrained_path)
    
    for epoch_idx in range(config.ae_epoch if config.train_autoencoder else 1):
        ae_stats = {'sum':0.0, 'cnt':0.0}
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
                sess.run(model.cf_optimize, feed_dict=feed_dict)

            if (datum_idx % (batch_size*5) == 0) or (datum_idx + batch_size >= len(inputs)):
                decoded, ae_loss, cf_acc = sess.run([model.decoded, model.ae_loss, model.cf_acc],
                        feed_dict=feed_dict)
                decoded = decoded.reshape((len(batch_inputs), config.max_time_step, config.input_dim))
                decoded_name = ''.join([idx2char[char] 
                    for char in np.argmax(decoded[0], 1)])[:batch_input_len[0]]
                original_name = ''.join([idx2char[char] 
                    for char in batch_inputs[0]])[:batch_input_len[0]]

                ae_stats['sum'] += ae_loss
                ae_stats['cnt'] += 1
                _progress = "\rEp %d: %s/%s, ae_loss: %.3f, cf_acc: %.3f" % (
                        epoch_idx, original_name, decoded_name, ae_loss, cf_acc)
                sys.stdout.write(_progress)
                sys.stdout.flush()

        ae_ep = ae_stats['sum'] / ae_stats['cnt']
        if ae_ep <= 0.010:
            print('\nAutoencoder Training Done with %.3f loss' % ae_ep)
            break
        else:
            print(' avg_loss: %.3f' % ae_ep)

    if config.train_autoencoder is True:
        model.save(config.checkpoint_dir)


    print('\n## GAN Training')
    d_iter = 1
    g_iter = 2 
    for epoch_idx in range(config.gan_epoch):
        # Initialize result file
        f = open(config.results_dir + '/' + model.scope, 'w')
        f.write('Results of Name Generation\n')
        f.close()
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
                d_loss, g_loss, cf_loss, g_decoded = sess.run(
                        [model.d_loss, model.g_loss, model.cf_loss_fake, model.g_decoded], 
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
                _progress = "\rEp %d d:%.3f, g:%.3f, cf:%.3f, %s (%s)" % \
                        (epoch_idx, d_loss, g_loss, cf_loss, g_decoded_name[:PAD_idx], 
                                idx2country[batch_labels[0]])
                sys.stdout.write(_progress)
                sys.stdout.flush()

            f = open(config.results_dir + '/' + model.scope, 'a')
            for decoded, label in zip(g_decoded, batch_labels):
                name = ''.join([idx2char[char] for char in np.argmax(decoded, 1)])
                if char2idx['PAD'] in np.argmax(decoded, 1):
                    PAD_idx = np.argwhere(np.argmax(decoded, 1) == char2idx['PAD'])
                    PAD_idx = PAD_idx.flatten().tolist()[0]
                else:
                    PAD_idx = -1
                f.write(name[:PAD_idx] + '\t' + idx2country[label] + '\n')
            f.close()
     
        print()

    model.save(config.checkpoint_dir)

