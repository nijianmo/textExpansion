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
from model import EncoderRNN, LuongAttnDecoderRNN, AttributeEncoder, Attn, AttributeAttn
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

def train(attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len, encoder1, encoder2, encoder3, decoder, embedding, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    encoder3_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    if USE_CUDA:
        attr_input = attr_input.cuda() 
        summary_input = summary_input.cuda()
        title_input = title_input.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []

    encoder_out1, encoder_out2, encoder_hidden = encoder3(summary_input, summary_input_lengths, title_input, title_input_lengths, None) # summary encoder
    encoder_out3, encoder1_hidden = encoder1(attr_input) # attribute encoder
    encoder_out4, encoder2_hidden = encoder2(attr_input) # aspect encoder

    decoder_input = target_variable[:-1]
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder1_hidden[:decoder.n_layers] + encoder2_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        decoder_output, decoder_hidden, decoder_attn1, decoder_attn2, decoder_attn3, gate = decoder(
            decoder_input, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4
        )
        mask_loss = masked_cross_entropy(decoder_output, target_variable[1:], mask[1:])
        loss += mask_loss
        print_losses.append(mask_loss.data[0])
    
    loss.backward()

    clip = 5.0
    ec1 = torch.nn.utils.clip_grad_norm(encoder1.parameters(), clip)
    ec2 = torch.nn.utils.clip_grad_norm(encoder2.parameters(), clip)
    ec3 = torch.nn.utils.clip_grad_norm(encoder3.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder1_optimizer.step()
    encoder2_optimizer.step()
    encoder3_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) 

def evaluate(attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len, encoder1, encoder2, encoder3, decoder, embedding, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder1.eval()
    encoder2.eval()
    encoder3.eval()
    decoder.eval()

    if USE_CUDA:
        attr_input = attr_input.cuda() 
        summary_input = summary_input.cuda()
        title_input = title_input.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []

    encoder_out1, encoder_out2, encoder_hidden = encoder3(summary_input, summary_input_lengths, title_input, title_input_lengths, None) # summary encoder
    encoder_out3, encoder1_hidden = encoder1(attr_input) # attribute encoder
    encoder_out4, encoder2_hidden = encoder2(attr_input) # aspect encoder

    decoder_input = target_variable[:-1]
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder1_hidden[:decoder.n_layers] + encoder2_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        decoder_output, decoder_hidden, decoder_attn1, decoder_attn2, decoder_attn3, gate = decoder(
            decoder_input, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4
        )
        mask_loss = masked_cross_entropy(decoder_output, target_variable[1:], mask[1:])
        loss += mask_loss
        print_losses.append(mask_loss.data[0])

    return sum(print_losses) 

def batchify(pairs, bsz, voc, reverse, evaluation=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(pairs) // bsz
    data = []
    for i in range(nbatch):
        data.append(batch2TrainData(voc, pairs[i * bsz: i * bsz + bsz], reverse, evaluation))
    return data

def trainIters(corpus, reverse, n_epoch, learning_rate, batch_size, n_layers, hidden_size, 
                print_every, loadFilename=None, attn_model='dot', decoder_learning_ratio=1.0):
    print("corpus: {}, reverse={}, n_epoch={}, learning_rate={}, batch_size={}, n_layers={}, hidden_size={}, decoder_learning_ratio={}".format(corpus, reverse, n_epoch, learning_rate, batch_size, n_layers, hidden_size, decoder_learning_ratio))

    voc, pairs, valid_pairs, test_pairs = loadPrepareData(corpus)
    print('load data...')

    path = "data/expansion"
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
    # aspect
    with open(os.path.join(save_dir, '15_aspect.pkl'), 'rb') as fp:
        aspect_ids = pickle.load(fp)
    aspect_num = 15 # 15 | 20 main aspects and each of them has 100 words
    aspect_ids = Variable(torch.LongTensor(aspect_ids), requires_grad=False) # convert list into torch Variable, used to index word embedding
    # attribute embeddings
    attr_size = 64 # 
    attr_num = 2

    print("corpus: {}, reverse={}, n_words={}, n_epoch={}, learning_rate={}, batch_size={}, n_layers={}, hidden_size={}, decoder_learning_ratio={}, attr_size={}, aspect_num={}".format(corpus, reverse, voc.n_words, n_epoch, learning_rate, batch_size, n_layers, hidden_size, decoder_learning_ratio, attr_size, aspect_num))
    with open(os.path.join(save_dir, 'user_item.pkl'), 'rb') as fp:
        user_dict, item_dict = pickle.load(fp)
    num_user = len(user_dict)
    num_item = len(item_dict)
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))
    attr_embeddings.append(nn.Embedding(num_item, attr_size))
    aspect_embeddings = []
    aspect_embeddings.append(nn.Embedding(num_user, aspect_num))
    aspect_embeddings.append(nn.Embedding(num_item, aspect_num))
    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()
        for aspect_embedding in aspect_embeddings:
            aspect_embedding = aspect_embedding.cuda()
        aspect_ids = aspect_ids.cuda()

    encoder1 = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)
    encoder2 = AttributeEncoder(aspect_num, attr_num, hidden_size, aspect_embeddings, n_layers)
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder3 = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, attr_size, voc.n_words, aspect_ids, n_layers)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder1.load_state_dict(checkpoint['en1'])
        encoder2.load_state_dict(checkpoint['en2'])
        encoder3.load_state_dict(checkpoint['en3'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    if USE_CUDA:
        encoder1 = encoder1.cuda()
        encoder2 = encoder2.cuda()
        encoder3 = encoder3.cuda()
        decoder = decoder.cuda()

    # optimizer
    print('Building optimizers ...')
    encoder1_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
    encoder2_optimizer = optim.Adam(encoder2.parameters(), lr=learning_rate)
    encoder3_optimizer = optim.Adam(encoder3.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder1_optimizer.load_state_dict(checkpoint['en1_opt'])
        encoder2_optimizer.load_state_dict(checkpoint['en2_opt'])
        encoder3_optimizer.load_state_dict(checkpoint['en3_opt'])
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
        encoder1.train()
        encoder2.train()
        encoder3.train()
        decoder.train()
        print_loss = 0
        start_time = time.time()
        for batch, training_batch in enumerate(training_batches):
            attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len = training_batch

            loss = train(attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len, encoder1, encoder2, encoder3, decoder, embedding, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer, batch_size)
            print_loss += loss
            perplexity.append(loss)
            #print("batch {} loss={}".format(batch, loss))
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
            attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len = val_batch
            loss = evaluate(attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len, encoder1, encoder2, encoder3, decoder, embedding, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer, batch_size)
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
                'en1': encoder1.state_dict(),
                'en2': encoder2.state_dict(),
                'en3': encoder3.state_dict(),
                'de': decoder.state_dict(),
                'en1_opt': encoder1_optimizer.state_dict(),
                'en2_opt': encoder2_optimizer.state_dict(),
                'en3_opt': encoder3_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, filename(reverse, 'lexicon_title_expansion_model'))))
            best_val_loss = val_loss
     
            # Run on test data.
            test_loss = 0
            for test_batch in test_batches:
                attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len = test_batch
                loss = evaluate(attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, target_variable, mask, max_target_len, encoder1, encoder2, encoder3, decoder, embedding, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer, batch_size)
                test_loss += loss
            test_loss /= len(test_batches)
            print('-' * 89)
            print('| test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
            print('-' * 89)

        if val_loss > best_val_loss:
            break

