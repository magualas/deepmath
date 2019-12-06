import torch
import torch.nn as nn
import numpy as np
from utils import *

class biRNN(nn.Module):
    """"
    Module for bidirectional RNN.
    """"
    def __init__(self, vocab_size, emb_size, rnn_size, rnn_layers, output_size, dropout, term_embeddings):
        super(biRNN, self).__init__()
        """
        vocab_size: number of symbols in the vocabulary of the model, vocab_ls
        emb_size: size of term (goal and hypotheses) embeddings
        rnn_size: size of hidden vectors of bidirectional RNN
        rnn_layers: number of layers in RNN
        output_size: 
        dropout: dropout (numerical)
        term_embeddings: precomputed embeddings of terms
        """
        
        # embedding layer - precomputed
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.embeddings.weight = nn.Parameter(term_embeddings, requires_grad=False)
        
        # bidirectional LSTM
        self.lstm = nn.LSTM(input_size = emb_size,
                            hidden_size = rnn_size,
                            num_layers = rnn_layers,
                            dropout = dropout,
                            bidirectional = True)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # fully connected layer
        self.fc = nn.Linear(rnn_size * rnn_layers * 2, output_size)
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()
    
    def forward(self, x):