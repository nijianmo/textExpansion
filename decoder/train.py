# raise ValueError("deal with Variable requires_grad, and .cuda()")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, DecoderRNN, LuongAttnDecoderRNN, Attn
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from util import *
import time
from masked_cross_entropy import * 
cudnn.benchmark = True

#############################################
# Training
#############################################

def maskNLLLoss(input, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda() if USE_CUDA else loss
    return loss, nTotal.data[0]

criterion = nn.CrossEntropyLoss(ignore_index = 0)

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, 
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    decoder_optimizer.zero_grad()

    if USE_CUDA:
        #input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []
    #n_totals = 0 

    #decoder_input = Variable(torch.LongTensor([[SOS_token for _ in range(batch_size)]]))
    decoder_input = target_variable[0].view(1, -1)
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = decoder.init_hidden(batch_size)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    '''
    if use_teacher_forcing:
        for t in range(1, max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0] * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            topv, topi = decoder_output.data.topk(1) # [64, 1]

            decoder_input = Variable(torch.LongTensor([[topi[i][0] for i in range(batch_size)]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0] * nTotal)
            n_totals += nTotal
    '''
    if use_teacher_forcing:
        for t in range(1, max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            #mask_loss = masked_cross_entropy(decoder_output, target_variable[t], mask[t])
            mask_loss = criterion(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0])
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            topv, topi = decoder_output.data.topk(1) # [64, 1]

            decoder_input = Variable(torch.LongTensor([[topi[i][0] for i in range(batch_size)]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            #mask_loss = masked_cross_entropy(decoder_output, target_variable[t], mask[t])
            mask_loss = criterion(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0])

    loss.backward()

    clip = 5.0
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    decoder_optimizer.step()

    return sum(print_losses) / (max_target_len - 1) 

def evaluate(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, 
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):
    
    decoder.eval()

    if USE_CUDA:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []

    #decoder_input = Variable(torch.LongTensor([[SOS_token for _ in range(batch_size)]]))
    decoder_input = target_variable[0].view(1, -1)
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = decoder.init_hidden(batch_size)

    use_teacher_forcing = True # evaluation always use teacher forcing

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(1, max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            #mask_loss = masked_cross_entropy(decoder_output, target_variable[t], mask[t])
            mask_loss = criterion(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0])
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            topv, topi = decoder_output.data.topk(1) # [64, 1]

            decoder_input = Variable(torch.LongTensor([[topi[i][0] for i in range(batch_size)]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            #mask_loss = masked_cross_entropy(decoder_output, target_variable[t], mask[t])
            mask_loss = criterion(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0])

    return sum(print_losses) / (max_target_len - 1) 


def batchify(pairs, bsz, voc, reverse, evaluation=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(pairs) // bsz
    data = []
    for i in range(nbatch):
        data.append(batch2TrainData(voc, pairs[i * bsz: i * bsz + bsz], reverse, evaluation))
    return data

def trainIters(corpus, reverse, n_epoch, learning_rate, batch_size, n_layers, hidden_size, 
                print_every, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):
    print("corpus: {}, reverse={}, n_epoch={}, learning_rate={}, batch_size={}, n_layers={}, hidden_size={}, decoder_learning_ratio={}".format(corpus, reverse, n_epoch, learning_rate, batch_size, n_layers, hidden_size, decoder_learning_ratio))

    voc, pairs, valid_pairs, test_pairs = loadPrepareData(corpus)
    print('load data...')

    path = "data/attr2seq"
    # training data
    corpus_name = corpus
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, path, '{}_{}.tar'.format(filename(reverse, 'training_batches'), batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = batchify(pairs, batch_size, voc, reverse)
        print('Complete building training pairs ...')
        torch.save(training_batches, os.path.join(save_dir, path, '{}_{}.tar'.format(filename(reverse, 'training_batches'), batch_size)))

    # validation/test data
    eval_batch_size = 10
    try:
        val_batches = torch.load(os.path.join(save_dir, path, '{}_{}.tar'.format(filename(reverse, 'val_batches'), eval_batch_size)))
    except FileNotFoundError:
        print('Validation pairs not found, generating ...')
        val_batches = batchify(valid_pairs, eval_batch_size, voc, reverse, evaluation=True)
        print('Complete building validation pairs ...')
        torch.save(val_batches, os.path.join(save_dir, path, '{}_{}.tar'.format(filename(reverse, 'val_batches'), eval_batch_size)))

    try:
        test_batches = torch.load(os.path.join(save_dir, path, '{}_{}.tar'.format(filename(reverse, 'test_batches'), eval_batch_size)))
    except FileNotFoundError:
        print('Test pairs not found, generating ...')
        test_batches = batchify(test_pairs, eval_batch_size, voc, reverse, evaluation=True)
        print('Complete building test pairs ...')
        torch.save(test_batches, os.path.join(save_dir, path, '{}_{}.tar'.format(filename(reverse, 'test_batches'), eval_batch_size)))

    # model
    checkpoint = None 
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = DecoderRNN(embedding, hidden_size, voc.n_words, n_layers)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_epoch = 0
    perplexity = []
    best_val_loss = None
    print_loss = 0
    if loadFilename:
        start_epoch = checkpoint['epoch'] + 1
        perplexity = checkpoint['plt']

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        # train epoch
        encoder.train()
        decoder.train()
        print_loss = 0
        start_time = time.time()
        for batch, training_batch in enumerate(training_batches):
            input_variable_attr, input_variable, lengths, target_variable, mask, max_target_len = training_batch

            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
            print_loss += loss
            perplexity.append(loss)
            #print("batch{} loss={}".format(batch, loss))
            if batch % print_every == 0 and batch > 0:
                cur_loss = print_loss / print_every
                elapsed = time.time() - start_time

                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(training_batches), learning_rate,
                        elapsed * 1000 / print_every, cur_loss, math.exp(cur_loss)))

                print_loss = 0
                start_time = time.time()
        # evaluate
        val_loss = 0
        for val_batch in val_batches:
            input_variable_attr, input_variable, lengths, target_variable, mask, max_target_len = val_batch
            loss = evaluate(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, eval_batch_size)
            val_loss += loss
        val_loss /= len(val_batches)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            directory = os.path.join(save_dir, 'model', '{}_{}'.format(n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, filename(reverse, 'text_decoder_model'))))
            best_val_loss = val_loss

            # Run on test data.
            test_loss = 0
            for test_batch in test_batches:
                input_variable_attr, input_variable, lengths, target_variable, mask, max_target_len = test_batch
                loss = evaluate(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                           decoder, embedding, encoder_optimizer, decoder_optimizer, eval_batch_size)
                test_loss += loss
            test_loss /= len(test_batches)
            print('-' * 89)
            print('| test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
            print('-' * 89)

        if val_loss > best_val_loss:
            break


