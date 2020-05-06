
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn


class TextRNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, hidden_size, num_hidden_layers, output_size, bidirectional,
                 dropout_rate, optimizer, learning_rate):

        super(TextRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)
        
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_hidden_layers,
                            dropout=self.dropout_rate,
                            bidirectional=self.bidirectional)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size * self.num_hidden_layers * (1 + self.bidirectional), self.output_size)

        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):

        # >> x: (max_sen_len, batch_size)
        embedded_sequence = self.embeddings(x)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)
        # >> embedded_sequence: (seq_len, batch_size, embedding_size)

        o_n, (h_n, c_n) = self.lstm(embedded_sequence)
        # >> h_n: (num_layers * num_directions, batch_size, hidden_size)
        feature_vec = self.dropout(h_n)
        feature_vec = torch.cat([feature_vec[i, :, :] for i in range(feature_vec.shape[0])], dim=1)
        # >> feature_vec: (batch_size, hidden_size * hidden_layers * num_directions) -> reshaping is for the linear layer

        logits = self.fc(feature_vec)
        preds = self.softmax(logits)

        return preds

