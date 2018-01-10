import torch
from torch.autograd import Variable
import random
from model import *
from util import *
from config import USE_CUDA
import sys
import os
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from masked_cross_entropy import *
import itertools
import random
import math
from tqdm import tqdm
from load import SOS_token, EOS_token, PAD_token, UNK_token
from model import EncoderRNN, LuongAttnDecoderRNN, Attn
import pickle
import logging
logging.basicConfig(level=logging.INFO)

 
class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        return sum(self.sentence_scores) / len(self.sentence_scores)
        # return mean of sentence_score

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)

        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.idx2word[idx] for idx in self.sentence_idxes] + ['<eos>'], 
                                   self.avgScore())) # tuple(word_list, score_float) 
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<eos>')
            else:
                words.append(voc.idx2word[self.sentence_idxes[i]])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<eos>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for t in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = Variable(torch.LongTensor([[sentence.last_idx]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.data.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=MAX_LENGTH):

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        topv, topi = decoder_output.data.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(voc.idx2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, max_length=MAX_LENGTH):
    #indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    #lengths = [len(indexes) for indexes in indexes_batch]
    #input_batch = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)

    input_batch = Variable(torch.LongTensor([sentence[:2]]), volatile=True)
    input_batch = input_batch.cuda() if USE_CUDA else input_batch

    encoder_outputs, encoder_hidden = encoder(input_batch)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, voc)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size)


def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
            print('>', pair[1])
        if beam_size == 1:
            output_words, attentions = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))

def evaluateInput(encoder, decoder, voc, beam_size):
    pair = ''
    while(1):
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, attentions = evaluate(encoder, decoder, voc, pair, beam_size)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")


def runTest(n_layers, hidden_size, reverse, modelFile, beam_size, input, corpus):

    voc, pairs, valid_pairs, test_pairs = loadPrepareData(corpus)
    
    print('Building encoder and decoder ...')
    # attribute embeddings
    attr_size = 64
    attr_num = 2
    with open(os.path.join(save_dir, 'user_item.pkl'), 'rb') as fp:
        user_dict, item_dict = pickle.load(fp)
    num_user = len(user_dict)
    num_item = len(item_dict)
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))    
    attr_embeddings.append(nn.Embedding(num_item, attr_size)) 
    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()
   
    encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)
    embedding = nn.Embedding(voc.n_words, hidden_size, padding_idx=0) # word embedding
    attn_model = 'concat'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, attr_size, voc.n_words, n_layers)    
    
    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    if input:
        evaluateInput(encoder, decoder, voc, beam_size)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, 30)
        #evaluateRandomly(encoder, decoder, voc, test_pairs, reverse, beam_size, 30)
