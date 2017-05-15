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

                    # if nationality != 'Japan':
                    #     continue
                    
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
    country_sorted = sorted(country2cnt.items(), key=operator.itemgetter(1))[::-1][0:10]
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

