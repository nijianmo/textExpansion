import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from config import USE_CUDA

# change forward function to accept one pair of input seqences rather than only one sequence
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    #def forward(self, input_seq, input_lengths, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        #embedded = self.embedding(input_seq)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        #outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        #return outputs, hidden

    def forward(self, x1, x1_len, x2, x2_len, hidden=None):
        '''
        :param x1, x2: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param x1_len, x2_len:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''

        # sort x2
        x2_sort_idx = np.argsort(-np.array(x2_len)).tolist()
        x2_unsort_idx = np.argsort(x2_sort_idx).tolist()
       
        x2_len = np.array(x2_len)[x2_sort_idx].tolist()
        x2 = x2[:, x2_sort_idx]
 
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        x1_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x1_emb, x1_len)
        x2_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x2_emb, x2_len)

        out1_p, hidden1 = self.gru(x1_emb_p, hidden) # output: (seq_len, batch, hidden*n_dir)
        out2_p, hidden2 = self.gru(x2_emb_p, hidden) # output: (seq_len, batch, hidden*n_dir)

        out1, out1_len = torch.nn.utils.rnn.pad_packed_sequence(out1_p)
        out2, out2_len = torch.nn.utils.rnn.pad_packed_sequence(out2_p)

        # unsort x2
        # print("x2_unsort_idx ", x2_unsort_idx)

        out2 = out2[:, x2_unsort_idx, :] # index batch axis

        out1 = out1[:, :, :self.hidden_size] + out1[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        out2 = out2[:, :, :self.hidden_size] + out2[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        hidden = hidden1 + hidden2
        
        return out1, out2, hidden

class AttributeEncoder(nn.Module):
    def __init__(self, attr_size, attr_num, hidden_size, attr_embeddings, n_layers=1, dropout=0.2):
        super(AttributeEncoder, self).__init__()
        self.attr_size = attr_size
        self.attr_num = attr_num
        self.hidden_size = hidden_size
        self.attr_embeddings = attr_embeddings
        self.n_layers = n_layers
        self.attr = nn.Linear(self.attr_size * self.attr_num, self.n_layers * self.hidden_size) # hidden matrix is L*n, where L is number of layers and n is hidden size of each unit
        self.tanh = nn.Tanh()

    def forward(self, input):
        embeddeds = [] # (attr_num, batch, attr_size) = (A,B,K)
        for i, attr_embedding in enumerate(self.attr_embeddings): # input[:, i] = (B, 1) for ith attribute, attr_embedding is the embedding layer for ith attribute
            embeddeds.append(attr_embedding(input[:, i]))

        embedded = torch.cat(embeddeds, 1) # (B,A*K)
        hidden = self.tanh(self.attr(embedded)) # (B, n_layers * hidden)
        hidden = hidden.view(-1, self.n_layers, self.hidden_size).transpose(0, 1).contiguous() # (B, n_layers, hidden) -> (n_layers, B, hidden)

        # embeddeds is attribute embeddings
        # hidden is the hidden variable to initialize decoder
        output = torch.stack(embeddeds) # (A,B,K) list to tenser
        return output, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        # end of update

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (N,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,N,T)
        '''
        max_len = encoder_outputs.size(0) # T
        seq_len = hidden.size(0) # N
        this_batch_size = encoder_outputs.size(1)
        
        H = hidden.repeat(max_len,1,1,1) # [T*N*B*H]
        encoder_outputs = encoder_outputs.repeat(seq_len,1,1,1).transpose(0,1) # [N*T*B*H] -> [T*N*B*H]
        #attn_energies = self.score(H,encoder_outputs) # compute attention score [B*N*T]
        attn_energies = F.tanh(self.score(H,encoder_outputs)) # compute attention score [B*N*T]
        return F.softmax(attn_energies, dim=2) # normalize with softmax on T axis

    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 3)) # [T*N*B*2H]->[T*N*B*H]
        energy = energy.view(-1, self.hidden_size) # [T*N*B,H]
        v = self.v.unsqueeze(1) #[1*H]
        #print(energy.size())
        #print(v.size())
        energy = energy.mm(v) # [T*N*B,H] * [H,1] -> [T*N*B,1]
        att_energies = energy.view(-1,hidden.size(1),hidden.size(2)) # [T*N*B] 
        att_energies = att_energies.transpose(0, 2).contiguous() # [B*N*T]
        return att_energies

class AttributeAttn(nn.Module):
    def __init__(self, method, hidden_size, attr_size):
        super(AttributeAttn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attr_size = attr_size
        self.attn = nn.Linear(self.hidden_size + attr_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        # end of update

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (N,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (A,B,K) = (# of attributes, batch size, attribute dimension
        :return
            attention energies in shape (B,N,A)
        '''
        attr_len = encoder_outputs.size(0) # A
        seq_len = hidden.size(0) # N
        this_batch_size = encoder_outputs.size(1)

        H = hidden.repeat(attr_len,1,1,1) # [A*N*B*H]
        encoder_outputs = encoder_outputs.repeat(seq_len,1,1,1).transpose(0,1).contiguous() # [N*A*B*H] -> [A*N*B*H]
        #attn_energies = self.score(H,encoder_outputs) # compute attention score [B*N*A]
        attn_energies = F.tanh(self.score(H,encoder_outputs)) # compute attention score [B*N*A]
        return F.softmax(attn_energies, dim=2) # normalize with softmax on A axis

    def score(self, hidden, encoder_outputs): # hidden (A,N,B,H)
        concat = torch.cat([hidden, encoder_outputs], 3)
        energy = self.attn(concat) # [A*N*B*(H+K)]->[A*N*B*H]
        energy = energy.view(-1, self.hidden_size) # [A*N*B,H]
        v = self.v.unsqueeze(1) #[1*H]
        energy = energy.mm(v) # [A*N*B,H] * [H,1] -> [A*N*B,1]
        att_energies = energy.view(-1,hidden.size(1),hidden.size(2)) # [A*N*B] 
        att_energies = att_energies.transpose(0, 2).contiguous() # [B*N*A]
        return att_energies


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.data.shape[0], -1) # (1,B,N)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N)->(B,N)
        context = context.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        # Return final output, hidden state
        return output, hidden

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, attr_size, output_size, aspect_ids, n_layers=1, dropout=0.2):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.aspect_ids = aspect_ids
        self.aspect_num = len(aspect_ids) // 100
        self.attr_size = attr_size

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 3 + attr_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(2*hidden_size + self.aspect_num, self.aspect_num)
        self.project = nn.Linear(2*self.aspect_num, self.aspect_num)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
        self.attr_attn = AttributeAttn(attn_model, hidden_size, attr_size)

    def forward(self, input_seq, last_hidden, encoder_out1, encoder_out2, encoder_out3, encoder_out4):
        # Note: we run all steps at one pass

        # Get the embedding of all input words
        '''
        :param input_seq:
            word input for all time steps, in shape (N*B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_out1: 
            encoder outputs from summary in shape (T*B*H)
        :param encoder_out2:
            encoder outputs from title in shape (S*B*H)
        :param encoder_out3:
            encoder outputs from attribute in shape (A*B*K)
        :param encoder_out4:
            encoder outputs in shape (A*B*P)
        '''
        batch_size = input_seq.size(1) # B
        seq_len = input_seq.size(0) # N
        embedded = self.embedding(input_seq) # [N*B*H]
        embedded = self.embedding_dropout(embedded)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden) 

        # Calculate attention
        attn_weights1 = self.attn(rnn_output, encoder_out1) # [N*B*H] x [T*B*H] -> [B*N*T]
        context1 = attn_weights1.bmm(encoder_out1.transpose(0, 1)) # [B*N*T] x [B*T*H] -> [B*N*H]
 
        attn_weights2 = self.attn(rnn_output, encoder_out2) # [N*B*H] x [S*B*H] -> [B*N*S]
        context2 = attn_weights2.bmm(encoder_out2.transpose(0, 1)) # [B*N*S] x [B*S*H] -> [B*N*H]

        attn_weights3 = self.attr_attn(rnn_output, encoder_out3) # [N*B*H] x [A*B*K] -> [B*N*A]
        context3 = attn_weights3.bmm(encoder_out3.transpose(0, 1).contiguous()) # [B*N*A] x [B*A*K] -> [B*N*K]

        context1 = context1.transpose(0, 1)        # [B*N*H] -> [N*B*H]
        context2 = context2.transpose(0, 1)        # [B*N*H] -> [N*B*H]
        context3 = context3.transpose(0, 1).contiguous()        # [B*N*K] -> [N*B*K]
   
        concat_input = torch.cat((rnn_output, context1, context2, context3), 2) # [N,B,3*H+K]
        concat_output = F.tanh(self.concat(concat_input)) 
        
        output = self.out(concat_output) # [N,B,W]

        output = output.clone()
        # Calculate aspect attention, define a gate to multiply the context
        encoder_out4 = torch.cat([encoder_out4[0], encoder_out4[1]],dim=-1) # [2,B,P] -> [B,2*P], A=2  
        encoder_out4 = self.project(encoder_out4) # [B,2*P] -> [B,P] 
        encoder_out4 = encoder_out4.repeat(seq_len,1,1)

        gate = F.tanh(self.gate(torch.cat((embedded, rnn_output, encoder_out4), dim=2))) # [N,B,(2H+P)] -> [N,B,P]         
        #gate = F.sigmoid(self.gate(torch.cat((embedded, rnn_output, encoder_out4), dim=2))) # [N,B,(2H+P)] -> [N,B,P]         
        #gate = F.tanh(self.gate(torch.cat((rnn_output, encoder_out4), dim=2))) # [N,B,(2H+P)] -> [N,B,P]         
        #gate = F.sigmoid(self.gate(torch.cat((embedded, rnn_output, encoder_out4), dim=2))) # [N,B,(2H+P)] -> [N,B,P]         

        output.scatter_add_(2, self.aspect_ids.repeat(seq_len, batch_size, 1), gate.repeat(1,1,100))
        #output = F.softmax(output, dim=2)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights1, attn_weights2, attn_weights3, gate

