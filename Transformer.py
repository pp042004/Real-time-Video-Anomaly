import torch.nn as nn
import torch
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(nn.Module):
  def __init__(self, embedding_dim, hidden_size, nheads, n_layers, max_len, num_labels, dropout):
    super(TransformerNet, self).__init__()
    # embedding layer
    # self.embedding = nn.Embedding(num_vocab, embedding_dim)
    
    # positional encoding layer
    self.pe = PositionalEncoding(embedding_dim, max_len = max_len)
    self.use_nested_tensor = True
    self.embedding_dim = embedding_dim
    self.max_len = max_len
    # encoder  layers
    enc_layer = nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout, batch_first=True)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)

    # final dense layer
    self.dense1 = nn.Linear(embedding_dim*max_len, embedding_dim)
    self.dense2 = nn.Linear(embedding_dim, num_labels)
    self.log_softmax = nn.LogSoftmax()
    self.features = False

  def forward(self, x):
    # x = self.embedding(x).permute(1, 0, 2)
    x = self.pe(x)
    # print(x.shape)
    x = self.encoder(x)
    # print(x.shape)
    x = x.reshape(x.shape[0], -1)
    x = self.dense1(x)
    features = x
    # print(x.shape)
    x = self.dense2(x)
    return features, x