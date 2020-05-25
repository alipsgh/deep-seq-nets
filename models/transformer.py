
import math
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        """
        PositionalEncoding module injects some information about the relative or absolute position
        of the tokens in the sequence. The positional encodings have the same dimension as the
        embeddings so that the two can be summed. Here, we use sine and cosine functions of
        different frequencies.
        :param d_model:
        :param dropout_rate:
        :param max_len:
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, dim_model, dim_ffn, num_heads, num_layers, output_size,
                 dropout_rate, optimizer, learning_rate):
        
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = dim_model
        self.num_heads = num_heads
        self.d_ffn = dim_ffn
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.output_size = output_size

        self.encoder = nn.Embedding(self.vocab_size, self.d_model)
        if embeddings is not None:
            self.encoder.weight = nn.Parameter(embeddings, requires_grad=False)

        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ffn, self.dropout_rate),
                                                         self.num_layers)

        self.fc = nn.Linear(self.d_model, self.output_size)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):

        x = x.t()
        x_e = self.encoder(x) * math.sqrt(self.d_model)
        x_pe = self.pos_encoder(x_e)

        e = self.transformer_encoder(x_pe)

        e = e[-1, :, :]
        logits = self.fc(e)

        return logits

