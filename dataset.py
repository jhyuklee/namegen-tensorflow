from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import time
import sys
import os
import operator

from random import shuffle
from utils import *

batch_size = 1000
data_dir = './data'
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
disp_epoch = 50
test_epoch = 200


def one_hot(index, length):
    vector = np.zeros([length])
    vector[index] = 1
    return vector


def get_name_data(data_dir):
    for root, dir, files in os.walk(data_dir):
        inputs = []
        inputs_length = []
        labels = []
        char_dict = {}
        country_dict = {}
        name_dict = {}
        name_max_len = 0
        collision_cnt = 0

        for file_cnt, file_name in enumerate(sorted(files)):
            data = open(os.path.join(root, file_name))
            file_len = 0
            
            if file_name == 'characters.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    char_dict[int(line[:-1].split('\t')[1])] = line.split('\t')[0]
            elif file_name == 'countries.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    country_dict[int(line[:-1].split('\t')[1])] = line.split('\t')[0]
            elif file_name == 'parsed.txt':
                for k, line in enumerate(data):
                    line = line[:-1]
                    name = [one_hot(int(k), 48) for k in line.split(']')[0][1:].split(', ')]
                    nationality = int(line.split(']')[1].split(' ')[1])
                    name_length = len(name)

                    if name_max_len < len(name):
                        name_max_len = len(name)
                    while len(name) != 50:
                        name.append(np.zeros([48]))
                   
                    name_string = ''.join([char_dict[char] for char in np.argmax(name, 1)][:name_length])
                    if name_string in name_dict:
                        collision_cnt += 1
                        name_dict[name_string] += 1
                        # print('collision cnt', str(collision_cnt), name_string[:len(name)])
                        continue
                    else:
                        name_dict[name_string] = 1

                    inputs.append(name)
                    labels.append(nationality)
                    inputs_length.append(name_length)
                    file_len = k + 1
            else:
                print('ignoring file', file_name)

            print('reading', file_name, 'of length', file_len)

    print('total data length:', len(inputs), len(labels), len(inputs_length))
    print('name max length:', name_max_len, 'to 50')
    print('unique name set:', len(name_dict))
    name_sorted = sorted(name_dict.items(), key=operator.itemgetter(1))
    print(name_sorted[::-1][:10])

    pairs = list(zip(inputs, inputs_length, labels))
    shuffle(pairs)
    inputs, inputs_length, labels = zip(*pairs)

    return np.array(inputs), np.array(labels), np.array(inputs_length), char_dict, country_dict


total_input, total_label, total_length, char_dict, country_dict = get_name_data(data_dir)
data_size = len(total_input)
train_input = total_input[:int(data_size * train_ratio)]
train_label = total_label[:int(data_size * train_ratio)]
train_length = total_length[:int(data_size * train_ratio)]

valid_input = total_input[int(data_size * train_ratio):int(data_size * (train_ratio + valid_ratio))]
valid_label = total_label[int(data_size * train_ratio):int(data_size * (train_ratio + valid_ratio))]
valid_length = total_length[int(data_size * train_ratio):int(data_size * (train_ratio + valid_ratio))]

test_input = total_input[int(data_size * (train_ratio + valid_ratio)):]
test_label = total_label[int(data_size * (train_ratio + valid_ratio)):]
test_length = total_length[int(data_size * (train_ratio + valid_ratio)):]

print('train:', train_input.shape, train_label.shape, train_length.shape, '\n' +
        'valid:', valid_input.shape, valid_label.shape, valid_length.shape, '\n' +
        'test:', test_input.shape, test_label.shape, test_length.shape)
print(train_input[0], train_label[0], train_length[0])
print(test_input[0], test_label[0], test_length[0])
print('preprocessing done\n')


def accuracy_score(labels, logits, logits_index, inputs=None):
    if logits_index is None:
        logits_per_step = logits
        labels_per_step = np.reshape(labels, -1)
    else:
        index = np.arange(0, len(labels)) * len(labels[0]) + (logits_index - 1)
        logits_per_step = logits[index]
        labels_per_step = np.reshape(labels, -1)[index]
        
        if inputs is not None:
            inputs = np.argmax(inputs, 2)
            f = open("./preds.txt", 'w')
            for logit, logit_index, label, input in zip(logits_per_step, logits_index, labels_per_step, inputs):
                name = ''.join([char_dict[char] for char in input][:logit_index])
                pred = 'pred => ' + str(np.argmax(logit)) + ':' + country_dict[np.argmax(logit)]
                corr = 'real => ' + str(label) + ':' + country_dict[label]
                result = '[correct]' if np.argmax(logit)== label else '[wrong]'
                end = '--------------------------------------------'
                # print(result + '\n' + name + '\n' + pred + '\n' + corr + '\n' + end + '\n')
                f.write(result + '\n' + name + '\n' + pred + '\n' + corr + '\n' + end + '\n')
            f.close()
     
    correct_prediction = np.equal(labels_per_step, np.argmax(logits_per_step, 1))
    accuracy = np.mean(correct_prediction.astype(float))
    return accuracy


def top_n_acc(labels, logits, logits_index, top, inputs=None):
    index = np.arange(0, len(labels)) * len(labels[0]) + (logits_index - 1)
    logits_per_step = logits[index]
    labels_per_step = np.reshape(labels, -1)[index]
        
    top_n_logits = [logit.argsort()[-top:][::-1] for logit in logits_per_step]
    correct_prediction = np.array([(pred in topn) for pred, topn in zip(labels_per_step,
        top_n_logits)])
    accuracy = np.mean(correct_prediction.astype(float))
    return accuracy


def train(model, config, sess):
    print('## Training')
    tf.global_variables_initializer().run()

    if config.continue_train is not False:
        model.load(config.checkpoint_dir)

    start_time = time.time()
    with tf.variable_scope('Result'):
        for epoch_idx in range(config.epoch):
            for datum_idx in range(0, len(train_input), batch_size):
                batch_inputs = train_input[datum_idx:datum_idx+batch_size]
                batch_input_len = train_length[datum_idx:datum_idx + batch_size]
                batch_labels = train_label[datum_idx:datum_idx+batch_size]

                feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                        model.labels: batch_labels}
                pred, _, cost, step, mask = sess.run([model.logits, model.optimize, 
                    model.losses, model.global_step, model.mask], feed_dict=feed_dict)

                if (datum_idx % (batch_size*5) == 0) \
                    or (datum_idx + batch_size >= len(train_input)):
                    acc = accuracy_score(batch_labels, pred, batch_input_len)
                    acc3 = top_n_acc(batch_labels, pred, batch_input_len, 3)
                    _progress = progress((datum_idx + batch_size) / float(len(train_input)))
                    _progress += " Training loss: %.3f, acc1: %.3f, acc3: %.3f, epoch: %d" % (cost,
                            acc, acc3, epoch_idx)
                    sys.stdout.write(_progress)
                    sys.stdout.flush()
            
            if epoch_idx % disp_epoch == 0 or epoch_idx == config.epoch - 1:
                valid_cost, valid_acc, valid_acc3 = test(model, config, sess, is_valid=True)
                print("\nValidation loss: %.3f, acc1: %.3f, acc3: %.3f" % (valid_cost, valid_acc,
                    valid_acc3))

            if epoch_idx % test_epoch == 0 or epoch_idx == config.epoch - 1:
                test_cost, test_acc, test_acc3 = test(model, config, sess, is_valid=False)
                print("Testing loss: %.3f, acc1: %.3f, acc3: %.3f" % (test_cost, test_acc,
                    test_acc3))
                model.save(config.checkpoint_dir, sess.run(model.global_step))
                print()

            summary = sess.run(model.merged_summary, feed_dict=feed_dict)
            model.train_writer.add_summary(summary, step)

    model.save(config.checkpoint_dir, sess.run(model.global_step))



def test(model, config, sess, is_valid=False):
    saved_dropout = model.output_dr
    model.output_dr = 1.0

    if is_valid:
        feed_dict = {model.inputs: valid_input, model.input_len: valid_length,
                model.labels: valid_label}
    else:
        feed_dict = {model.inputs: test_input, model.input_len: test_length,
                model.labels: test_label}
    
    pred, cost, step, summary = sess.run([model.logits, model.losses, model.global_step, model.merged_summary],
            feed_dict=feed_dict)

    # save pred, test_input, test_label
    
    if is_valid:
        acc = accuracy_score(valid_label, pred, valid_length, valid_input)
        acc3 = top_n_acc(valid_label, pred, valid_length, 3)
        model.valid_writer.add_summary(summary, step)
    else:
        acc = accuracy_score(test_label, pred, test_length)
        acc3 = top_n_acc(test_label, pred, test_length, 3)
        model.test_writer.add_summary(summary, step)

    model.output_dr = saved_dropout
    return cost, acc, acc3

