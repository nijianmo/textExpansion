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
from model import EncoderRNN, DecoderRNN, LuongAttnDecoderRNN, Attn
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
        #topi = topi.squeeze(0)
        #topv = topv.squeeze(0)
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

def beam_decode(decoder, decoder_hidden, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for t in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = Variable(torch.LongTensor([[sentence.last_idx]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            topv, topi = decoder_output.data.exp().topk(beam_size)
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

def decode(decoder, decoder_hidden, voc, max_length=MAX_LENGTH):

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden
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

    return decoded_words


def evaluate(encoder, decoder, voc, sentence, beam_size, max_length=MAX_LENGTH):
    sentence = sentence[2]
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)

    input_batch = input_batch.cuda() if USE_CUDA else input_batch

    decoder_hidden = decoder.init_hidden(1) # decoder do not use encoder information
    print("beam size={}".format(beam_size))
    if beam_size == 1:
        return decode(decoder, decoder_hidden, voc)
    else:
        return beam_decode(decoder, decoder_hidden, voc, beam_size)

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
            output_words = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))

def runTest(n_layers, hidden_size, reverse, modelFile, beam_size, input, corpus):

    voc, pairs, valid_pairs, test_pairs = loadPrepareData(corpus)
    
    print('Building encoder and decoder ...')
    '''# attribute embeddings
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
    '''
    embedding = nn.Embedding(voc.n_words, hidden_size, padding_idx=0) # word embedding
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'concat'
    decoder = DecoderRNN(embedding, hidden_size, voc.n_words, n_layers) 

    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
   
 
    evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, 2)
    #evaluateRandomly(encoder, decoder, voc, test_pairs, reverse, beam_size, 2)
    

    #sample(encoder, decoder, voc, pairs, reverse)
    #sample(encoder, decoder, voc, test_pairs, reverse)

def sample(encoder, decoder, voc, pairs, reverse):
    n_words = 100
    START_TOKEN = '<str>'
    STOP_TOKEN = '<eos>'
    word2idx = voc.word2idx
    idx2word = voc.idx2word

    path = "./metrics/"
    f1 = open(path + "ref.txt",'w')
    f2 = open(path + "tst.txt",'w')

    ref = []
    tst = []
    all_lens = []
    for n in range(100):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            if len(pair[1]) <= 2:
                continue
            print('>', pair[0])
            print('>', " ".join(pair[1]))
            sentence = " ".join(pair[1][1:-1])
            f1.write(sentence + "\n")
            ref.append(sentence)

        sentence = pair[0][2]
        indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
        lengths = [len(indexes) for indexes in indexes_batch]
        input_batch = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)

        input_batch = input_batch.cuda() if USE_CUDA else input_batch

        for temperature in [1]:
            print('sample {} temperature {}:'.format(n, temperature))
            hidden = decoder.init_hidden(1)
            word_idx = word2idx[START_TOKEN]
            input = Variable(torch.rand(1, 1).mul(word_idx).long(), volatile=True)
            if USE_CUDA:
                input = input.cuda()
            cnt = 0
            sentence = [START_TOKEN]
            for i in range(n_words):
                cnt += 1
                output, hidden = decoder(input, hidden)
                word_weights = output.squeeze().data.div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word = idx2word[word_idx]
                if word == STOP_TOKEN:
                    break
                if word == "<unk>":
                    sentence.append("unk")
                else:
                    sentence.append(word)
                input.data.fill_(word_idx)
            sentence = " ".join(sentence)
            print(sentence)
            f2.write(sentence + "\n")
            tst.append(sentence)
            all_lens.append(cnt)
    print('average length of generated samples is: {}'.format(sum(all_lens) / len(all_lens)))
    f1.close()
    f2.close()

    with open('generated.pkl','wb') as f:
        pickle.dump((ref,tst),f)


