import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(p=dropout)# 
        
    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)# 
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))# 
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)

        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs

#! The previous classes were not be changed from the my_network.py file
#! We need it here only to derive from them
#TODO: import this classes from the file

#! here we can see, that the code of the encoder is not changed
#! there is nothing special, cz all the "attention" is in the decoder
class AttentionEncoder(Encoder):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__(input_dim, emb_dim, hid_dim, n_layers, dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        
        return output, hidden, cell

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.softmax = nn.Softmax(dim=2)
        
        self.rnn = nn.LSTM(
            input_size=emb_dim+n_layers*hid_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.out = nn.Linear(hid_dim, output_dim)
                
    def forward(self, input, enc_hidden, dec_hidden, dec_cell):
 
        batch_size = dec_hidden.shape[1]
        input = input.unsqueeze(0)
        #! embedded is a self decoder context vector
        embedded = self.dropout(self.embedding(input))

        #! lets generate weighted vector of the context, using encoder hidden context vectors
        
        # take an encoder hidden vectors and multiple it by decoder input context vector
        hidd_dot_prod = torch.einsum("sbh, nbh -> bns", enc_hidden, dec_hidden) # long story short -- this function does a scalar multiplication; pattern "sbh, nbh -> bns" represents how the input dimentions have to be modified
        # produce a coefficients from the vectors multiplications
        hidd_coeff = self.softmax(hidd_dot_prod)
        # make a weighted additions to the decoder context from the encoder hidden context vectors
        attention_hidd = torch.einsum("sbh, bns -> bnh", enc_hidden, hidd_coeff)
        attention_hidd = attention_hidd.reshape(1, batch_size, -1)

        # produce a new decoder input vector, by concatenation it with weighted vectors of the encder
        modified_input = torch.cat([attention_hidd, embedded], dim=-1)

        output, (hidden, cell) = self.rnn(modified_input, (dec_hidden, dec_cell))
        prediction = self.out(output.squeeze(0))        
        return prediction, hidden, cell
    
#! Also we've no smth new in the seq2seq model itself
class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, encoder, decoder, device):
        super().__init__(encoder, decoder, device)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)       
        enc_output, dec_hidden, dec_cell = self.encoder(src)       
        input = trg[0,:]
        for t in range(1, max_len):
            output, dec_hidden, dec_cell = self.decoder(input, 
                                                        enc_output, 
                                                        dec_hidden, 
                                                        dec_cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(dim=1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs