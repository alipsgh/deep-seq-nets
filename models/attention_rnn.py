
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn
from torch.nn import functional as F


class AttentionRNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, hidden_size, num_hidden_layers, output_size, bidirectional,
                 dropout_rate, optimizer, learning_rate):

        super(AttentionRNN, self).__init__()

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
                            bidirectional=self.bidirectional)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size * (1 + self.bidirectional) * 2, self.output_size)

        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    @staticmethod
    def apply_attention(rnn_output, final_hidden_state):
        """
        Apply Attention on RNN output
        :param rnn_output: (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
        :param final_hidden_state: (batch_size, num_directions * hidden_size): final hidden state of the RNN
        :return:
        """
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)  # >> shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output
        
    def forward(self, x):

        embedded_sequence = self.embeddings(x)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)
        # >> embedded_sequence: (seq_len, batch_size, embedding_size)

        o_n, (h_n, c_n) = self.lstm(embedded_sequence)
        # >> o_n: (seq_len, batch_size, num_directions * hidden_size)
        # >> h_n: (num_directions, batch_size, hidden_size)

        batch_size = h_n.shape[1]
        final_h_n = h_n.view(self.num_hidden_layers, self.bidirectional + 1, batch_size, self.hidden_size)[-1, :, :, :]

        final_hidden_state = torch.cat([final_h_n[i, :, :] for i in range(final_h_n.shape[0])], dim=1)
        
        attention_out = self.apply_attention(o_n.permute(1, 0, 2), final_hidden_state)
        # >> attention_out: (batch_size, num_directions * hidden_size)

        feature_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        feature_vector = self.dropout(feature_vector)  # >> feature_vector: (batch_size, num_directions * hidden_size)
        logits = self.fc(feature_vector)

        return logits

