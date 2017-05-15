import tensorflow as tf
import numpy as np
import time
import random
import os
import sys


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
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_decoder_inputs), 'not the same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.labels: batch_labels, model.decoder_inputs: batch_decoder_inputs}
            
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
        if ae_ep <= 0.001:
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
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_z) == len(batch_decoder_inputs), 'not the same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.labels: batch_labels, model.decoder_inputs: batch_decoder_inputs}

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


def train_vae(model, dataset, config):
    sess = model.session
    batch_size = config.batch_size
    sets, char_set, country_set = dataset
    inputs, decoder_inputs, labels, inputs_length = sets
    idx2char, char2idx = char_set
    idx2country, country2idx = country_set

    print('\n## Variational Autoencoder Training')
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    
    for epoch_idx in range(config.vae_epoch):
        vae_stats = {'sum':0.0, 'cnt':0.0}
        for datum_idx in range(0, len(inputs), batch_size):
            batch_inputs = inputs[datum_idx:datum_idx+batch_size]
            batch_decoder_inputs = decoder_inputs[datum_idx:datum_idx+batch_size]
            batch_input_len = inputs_length[datum_idx:datum_idx + batch_size]
            batch_labels = labels[datum_idx:datum_idx+batch_size]
                
            assert len(batch_inputs) == len(batch_input_len) == len(batch_labels) == \
            len(batch_decoder_inputs), 'not the same batch size'

            feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                    model.labels: batch_labels, model.decoder_inputs: batch_decoder_inputs}
            
            sess.run(model.vae_optimize, feed_dict=feed_dict)

            if (datum_idx % (batch_size*5) == 0) or (datum_idx + batch_size >= len(inputs)):
                decoded, vae_loss = sess.run([model.decoded, model.vae_loss], feed_dict=feed_dict)
                decoded = decoded.reshape((len(batch_inputs), config.max_time_step, config.input_dim))
                decoded_name = ''.join([idx2char[char] 
                    for char in np.argmax(decoded[0], 1)])[:batch_input_len[0]]
                original_name = ''.join([idx2char[char] 
                    for char in batch_inputs[0]])[:batch_input_len[0]]

                vae_stats['sum'] += vae_loss
                vae_stats['cnt'] += 1
                _progress = "\rEp %d: %s/%s, vae_loss: %.3f" % (
                        epoch_idx, original_name, decoded_name, vae_loss)
                sys.stdout.write(_progress)
                sys.stdout.flush()
        print()

    print('\nSampling results')
    num_sample = 10
    batch_inputs = inputs[0:num_sample]
    batch_decoder_inputs = decoder_inputs[0:num_sample]
    batch_input_len = inputs_length[0:num_sample]
    batch_labels = labels[0:num_sample]
    batch_z = np.random.randn(num_sample, config.latent_dim)

    samples = []
    nation_list = ['Japan', 'France', 'USSR', 'Italy', 'Great Britain']
    decoded = model.sample(num_sample)
    for nation in nation_list:
        batch_cond = [country2idx[nation]] * num_sample
        feed_dict = {model.inputs: batch_inputs, model.input_len: batch_input_len, 
                model.labels: batch_labels, model.decoder_inputs: batch_decoder_inputs, 
                model.z: batch_z, model.cond: batch_cond}
        samples.append(sess.run(decoded, feed_dict=feed_dict))

    for results, nation in zip(samples, nation_list):
        print('[Results of', nation + ']')
        for sample in results:
            decoded = ''.join([idx2char[c] for c in sample])
            if 'PAD' in decoded:
                print(decoded[:decoded.index('PAD')], end=' / ')
            else:
                print(decoded, end=' ')
        print()

    model.save(config.checkpoint_dir)
 
