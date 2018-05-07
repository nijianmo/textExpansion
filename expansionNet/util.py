import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from masked_cross_entropy import *
import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token, UNK_token
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from config import MAX_LENGTH, save_dir
import pickle
import logging
logging.basicConfig(level=logging.INFO)

#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename

#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    ids = []
    for word in sentence:
        word = word.lower()
        if word in voc.word2idx:
            ids.append(voc.word2idx[word])
        else:
            ids.append(UNK_token)
    return ids
    # return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue)) 

def binaryMatrix(l, value=PAD_token):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1) # mask = 1 if not padding
    return m

# return attribute index and input pack_padded_sequence
def inputVar(data, voc, evaluation=False):
    with open(os.path.join(save_dir, 'word_vocab.pkl'), 'rb') as fp:
        word2idx, idx2word = pickle.load(fp)
    attr = [[d[0], d[1]] for d in data]
    summaryVar = [d[2] for d in data]
    titleVar = [d[3] for d in data]

    attrVar = Variable(torch.LongTensor(attr), volatile=evaluation) # (batch, attribute_num), in our case it is 2

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in summaryVar]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = Variable(torch.LongTensor(padList), volatile=evaluation)

    title_indexes_batch = [indexesFromSentence(voc, sentence) for sentence in titleVar]
    title_lengths = [len(indexes) for indexes in title_indexes_batch]
    title_padList = zeroPadding(title_indexes_batch)
    title_padVar = Variable(torch.LongTensor(title_padList), volatile=evaluation)

    return (attrVar, padVar, lengths, title_padVar, title_lengths) # attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = Variable(torch.ByteTensor(mask))
    padVar = Variable(torch.LongTensor(padList))
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by output length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, pair_batch, reverse, evaluation=False):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0][2]), reverse=True) # sort on summary length
    input_batch, output_batch = [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        output_batch.append(pair_batch[i][1])
    attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths = inputVar(input_batch, voc, evaluation=evaluation)
    output, mask, max_target_len = outputVar(output_batch, voc) # convert sentence to ids and padding
    return attr_input, summary_input, summary_input_lengths, title_input, title_input_lengths, output, mask, max_target_len

