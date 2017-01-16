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
vocab_size = 51
PAD = 48
GO = 49
EOS = 50
max_name_len = 50

data_dir = './data'
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
disp_epoch = 50
test_epoch = 200


def one_hot(index, length, value=1):
    assert index >= 0 and index < length, 'index must be bigger or equal than 0'
    vector = np.zeros([length])
    vector[index] = value
    return vector


def get_name_data(data_dir):
    for root, dir, files in os.walk(data_dir):
        inputs = []
        inputs_length = []
        decoder_inputs = []
        labels = []
        char_dict = {}
        country_dict = {}
        name_dict = {}
        max_len = 0
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
                    name = [one_hot(int(char), vocab_size) for char in line.split(']')[0][1:].split(', ')]
                    decoder_name = np.insert(name[:], 0, one_hot(GO, vocab_size), axis=0)
                    decoder_name = np.append(decoder_name[:], [one_hot(EOS, vocab_size)], axis=0)
                    nationality = one_hot(len(name)-1, max_name_len, int(line.split(']')[1].split(' ')[1]))
                    name_length = len(name)

                    if max_len < len(name): # update the maximum length
                        max_len = len(name)
                    while len(name) != max_name_len: # fill with PAD
                        name.append(one_hot(PAD, vocab_size))
                    while len(decoder_name) != max_name_len:
                        decoder_name = np.append(decoder_name[:], [one_hot(PAD, vocab_size)], axis=0)
                   
                    name_string = ''.join([char_dict[char] for char in np.argmax(name, 1)][:name_length])
                    if name_string in name_dict:
                        collision_cnt += 1
                        name_dict[name_string] += 1
                        # print('collision cnt', str(collision_cnt), name_string[:len(name)])
                        continue
                    else:
                        name_dict[name_string] = 1

                    inputs.append(name)
                    decoder_inputs.append(decoder_name)
                    labels.append(nationality)
                    inputs_length.append(name_length)
                    file_len = k + 1
            else:
                print('ignoring file', file_name)

            print('reading', file_name, 'of length', file_len)

    print('total data length:', len(inputs), len(labels), len(inputs_length))
    print('name max length:', max_len, '/', max_name_len)
    print('unique name set:', len(name_dict))
    name_sorted = sorted(name_dict.items(), key=operator.itemgetter(1))
    print(name_sorted[::-1][:10])

    pairs = list(zip(inputs, inputs_length, labels, decoder_inputs))
    shuffle(pairs)
    inputs, inputs_length, labels, decoder_inputs = zip(*pairs)

    return np.array(inputs), np.array(decoder_inputs), np.array(labels), np.array(inputs_length), char_dict, country_dict


total_input, total_decoder_input, total_label, total_length, char_dict, country_dict = get_name_data(data_dir)
data_size = len(total_input)

print('train:', total_input.shape, total_decoder_input.shape, total_label.shape, total_length.shape)
print(np.argmax(total_input[0], 1))
print(np.argmax(total_decoder_input[0], 1), np.argmax(total_label[0], 0), total_length[0])
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
    
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    if config.continue_train is not False:
        model.load(config.checkpoint_dir)

    start_time = time.time()

    for epoch_idx in range(config.ed_epoch):
        for datum_idx in range(0, len(total_input), batch_size):
            batch_inputs = total_input[datum_idx:datum_idx+batch_size]
            batch_decoder_inputs = total_decoder_input[datum_idx:datum_idx+batch_size]
            batch_input_len = total_length[datum_idx:datum_idx + batch_size]
            batch_labels = total_label[datum_idx:datum_idx+batch_size]
            batch_z = np.random.uniform(-1, 1, (len(batch_inputs), config.input_dim))
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_z) == len(batch_decoder_inputs), 'not same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.z: batch_z, model.labels: batch_labels, model.decoder_inputs:
                    batch_decoder_inputs}

            sess.run(model.ed_optimize, feed_dict=feed_dict)

            if (datum_idx % (batch_size*5) == 0) \
                or (datum_idx + batch_size >= len(total_input)):
                decoded, ed_loss = sess.run([model.decoded, model.ed_loss], feed_dict=feed_dict)
                decoded = decoded.reshape((len(batch_inputs), config.max_time_step,
                    config.input_dim))
                decoded_name = ''.join([char_dict[char] for char in np.argmax(decoded[0], 1)])[:batch_input_len[0]]
                original_name = ''.join([char_dict[char] for char in np.argmax(batch_inputs[0], 1)])[:batch_input_len[0]]
                _progress = progress((datum_idx + batch_size) / float(len(total_input)))
                _progress += " Training decoded: %s/%s, ed_loss: %.3f, epoch: %d" % (original_name,
                        decoded_name, ed_loss, epoch_idx)
                sys.stdout.write(_progress)
                sys.stdout.flush()

        print()


    for epoch_idx in range(config.gan_epoch):
        for datum_idx in range(0, len(total_input), batch_size):
            batch_inputs = total_input[datum_idx:datum_idx+batch_size]
            batch_decoder_inputs = total_decoder_input[datum_idx:datum_idx+batch_size]
            batch_input_len = total_length[datum_idx:datum_idx + batch_size]
            batch_labels = total_label[datum_idx:datum_idx+batch_size]
            batch_z = np.random.uniform(-1, 1, (len(batch_inputs), config.input_dim))
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_z) == len(batch_decoder_inputs), 'not same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.z: batch_z, model.labels: batch_labels, model.decoder_inputs:
                    batch_decoder_inputs}

            sess.run([model.d_optimize_real, model.d_optimize_fake], feed_dict=feed_dict)
            sess.run(model.g_optimize, feed_dict=feed_dict)
            sess.run(model.g_optimize, feed_dict=feed_dict)
            sess.run(model.g_optimize, feed_dict=feed_dict)

            if (datum_idx % (batch_size*5) == 0) \
                or (datum_idx + batch_size >= len(total_input)):
                d_loss, g_loss, g_decoded = sess.run([model.d_loss, model.g_loss, model.g_decoded], feed_dict=feed_dict)
                g_decoded = g_decoded.reshape((len(batch_inputs), config.max_time_step,
                    config.input_dim))
                g_decoded_name = ''.join([char_dict[char] for char in np.argmax(g_decoded[0], 1)])
                if PAD in np.argmax(g_decoded[0], 1):
                    PAD_idx = np.argwhere(np.argmax(g_decoded[0], 1) == PAD)[0]
                else:
                    PAD_idx = -1
                _progress = progress((datum_idx + batch_size) / float(len(total_input)))
                _progress += " Training d_loss: %.3f, g_loss: %.3f, g_decoded: %s, epoch: %d" % \
                        (d_loss, g_loss, g_decoded_name[:PAD_idx], epoch_idx)
                sys.stdout.write(_progress)
                sys.stdout.flush()

        if epoch_idx % test_epoch == 0 or epoch_idx == config.gan_epoch - 1:
            #test_cost, test_acc, test_acc3 = test(model, config, sess, is_valid=False)
            #print("Testing loss: %.3f, acc1: %.3f, acc3: %.3f" % (test_cost, test_acc,
            #    test_acc3))
            #model.save(config.checkpoint_dir, sess.run(model.global_step))
            #print()
            pass

        # summary = sess.run(model.merged_summary, feed_dict=feed_dict)
        # model.train_writer.add_summary(summary, step)
        print()

    # model.save(config.checkpoint_dir, sess.run(model.global_step))


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

