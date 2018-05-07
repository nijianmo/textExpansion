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
from model import EncoderRNN, LuongAttnDecoderRNN, AttributeEncoder, Attn, AttributeAttn
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

def beam_decode(decoder, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for t in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = Variable(torch.LongTensor([[sentence.last_idx]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            decoder_output, decoder_hidden, attn1, attn2, attn3, gate = decoder(decoder_input, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4)

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

def decode(decoder, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4, voc, max_length=MAX_LENGTH):

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden, attn1, attn2, attn3, gate = decoder(decoder_input, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4)

        topv, topi = decoder_output.data.topk(3)
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(voc.idx2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words


def evaluate(encoder1, encoder2, encoder3, decoder, voc, pair, beam_size, max_length=MAX_LENGTH):
    sentence = pair[:2] # (user_id, item_id)
    attr_input = Variable(torch.LongTensor([sentence]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    sentence = pair[2] # summary
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    summary_input_lengths = [len(indexes) for indexes in indexes_batch]
    summary_input = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)
    summary_input = summary_input.cuda() if USE_CUDA else input_batch

    sentence = pair[3] # title
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    title_input_lengths = [len(indexes) for indexes in indexes_batch]
    title_input = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)
    title_input = title_input.cuda() if USE_CUDA else input_batch

    encoder_out1, encoder_out2, encoder_hidden = encoder3(summary_input, summary_input_lengths, title_input, title_input_lengths, None) # summary encoder
    encoder_out3, encoder1_hidden = encoder1(attr_input) # attribute encoder
    encoder_out4, encoder2_hidden = encoder2(attr_input) # aspect encoder

    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder1_hidden[:decoder.n_layers] + encoder2_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4, voc)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4, voc, beam_size)


def evaluateRandomly(encoder1, encoder2, encoder3, decoder, voc, pairs, reverse, beam_size, n=10):
    path = "./metrics/"
    f1 = open(path + "ref.txt",'w')
    f2 = open(path + "tst.txt",'w')
    for i in range(n):
        pair = pairs[i]
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
            print('>', pair[1])

        f1.write(" ".join(pair[1][1:-1]) + "\n")
        if beam_size == 1:
            output_words = evaluate(encoder1, encoder2, encoder3, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            f2.write(" ".join(output_words[:-1]) + "\n")
        else:
            output_words_list = evaluate(encoder1, encoder2, encoder3, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))
    f1.close()
    f2.close()



def runTest(n_layers, hidden_size, reverse, modelFile, beam_size, input, corpus):

    voc, pairs, valid_pairs, test_pairs = loadPrepareData(corpus)
    
    print('Building encoder and decoder ...')
    # aspect
    with open(os.path.join(save_dir, '15_aspect.pkl'), 'rb') as fp:
        aspect_ids = pickle.load(fp)
    aspect_num = 15 # 15 | 20 main aspects and each of them has 100 words
    aspect_ids = Variable(torch.LongTensor(aspect_ids), requires_grad=False) # convert list into torch Variable, used to index word embedding
    # attribute embeddings
    attr_size = 64 # 
    attr_num = 2

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
    checkpoint = torch.load(modelFile)
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

    # train mode set to false, effect only on dropout, batchNorm
    encoder1.train(False);
    encoder2.train(False);
    encoder3.train(False);
    decoder.train(False);

    #evaluateRandomly(encoder1, encoder2, encoder3, decoder, voc, pairs, reverse, beam_size, 100)
    evaluateRandomly(encoder1, encoder2, encoder3, decoder, voc, test_pairs, reverse, beam_size, len(test_pairs))
   
    #sample(encoder1, encoder2, decoder, voc, pairs, reverse)
    #sample(encoder1, encoder2, decoder, voc, test_pairs, reverse)


def sample(encoder1, encoder2, decoder, voc, pairs, reverse):
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
            print('>', pair[0])
            print('>', " ".join(pair[1]))
            sentence = " ".join(pair[1][1:-1])
            f1.write(sentence + "\n")
            ref.append(sentence)

        sentence = pair[0][:2] # (user_id, item_id)
        attr_input = Variable(torch.LongTensor([sentence]), volatile=True)
        attr_input = attr_input.cuda() if USE_CUDA else attr_input

        sentence = pair[0][2] # summary
        indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
        summary_input_lengths = [len(indexes) for indexes in indexes_batch]
        summary_input = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)
        summary_input = summary_input.cuda() if USE_CUDA else input_batch

        sentence = pair[0][3] # title
        indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
        title_input_lengths = [len(indexes) for indexes in indexes_batch]
        title_input = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)
        title_input = title_input.cuda() if USE_CUDA else input_batch

        encoder_out1, encoder_out2, encoder_hidden = encoder2(summary_input, summary_input_lengths, title_input, title_input_lengths, None)
        encoder_out3, encoder_hidden2 = encoder1(attr_input)

        for temperature in [1]:
            #hidden = encoder_hidden[:decoder.n_layers]
            hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden2[:decoder.n_layers]

            word_idx = word2idx[START_TOKEN]
            input = Variable(torch.rand(1, 1).mul(word_idx).long(), volatile=True)
            if USE_CUDA:
                input = input.cuda()
            cnt = 0
            sentence = []
            for i in range(n_words):
                cnt += 1
                #output, hidden, attn1, attn2 = decoder(input, hidden, encoder_out1, encoder_out2)
                output, hidden, attn1, attn2, attn3 = decoder(input, hidden, encoder_out1, encoder_out2, encoder_out3)

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

