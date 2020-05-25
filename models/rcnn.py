
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn
from torch.nn import functional as F


class RCNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, num_hidden_layers, hidden_size, linear_size, output_size,
                 dropout_rate, optimizer, learning_rate):

        super(RCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        # BiLSTM
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_hidden_layers,
                            dropout=self.dropout_rate, bidirectional=True)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.linear = nn.Sequential(nn.Linear(self.embedding_size + 2 * self.hidden_size, self.linear_size),
                                    nn.Tanh())
        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc = nn.Linear(self.linear_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, x):

        embedded_sequence = self.embeddings(x)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)  # >> embedded_sequence: (seq_len, batch_size, embed_size)

        o_n, (_, _) = self.lstm(embedded_sequence)  # >> o_n: (seq_len, batch_size, 2 * hidden_size)
        input_features = torch.cat([o_n, embedded_sequence], 2).permute(1, 0, 2)
        # >> input_features: (batch_size, seq_len, embed_size + 2 * hidden_size)
        
        linear_output = self.linear(input_features)  # >> linear_output: (batch_size, seq_len, hidden_size_linear)
        linear_output = linear_output.permute(0, 2, 1)  # >> Reshaping for max_pool
        
        out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)  # >> out_features: (batch_size, hidden_size_linear)
        out_features = self.dropout(out_features)

        logits = self.fc(out_features)

        return logits

