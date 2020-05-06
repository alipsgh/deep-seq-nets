
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn


class CharCNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, num_channels, linear_size, seq_len, output_size,
                 dropout_rate, optimizer, learning_rate):

        super(CharCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = vocab_size
        self.num_channels = num_channels
        self.linear_size = linear_size
        self.seq_len = seq_len
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        conv_1 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels, kernel_size=7),
                               nn.ReLU(), nn.MaxPool1d(kernel_size=3))  # >> (batch_size, num_channels, (seq_len-6)/3)
        conv_2 = nn.Sequential(nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=7),
                               nn.ReLU(), nn.MaxPool1d(kernel_size=3))  # >> (batch_size, num_channels, (seq_len-6-18)/(3*3))
        conv_3 = nn.Sequential(nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3),
                               nn.ReLU())  # >> (batch_size, num_channels, (seq_len-6-18-18)/(3*3))
        conv_4 = nn.Sequential(nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3),
                               nn.ReLU())  # >> (batch_size, num_channels, (seq_len-6-18-18-18)/(3*3))
        conv_5 = nn.Sequential(nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3),
                               nn.ReLU())  # >> (batch_size, num_channels, (seq_len-6-18-18-18-18)/(3*3))
        conv_6 = nn.Sequential(nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3),
                               nn.ReLU(), nn.MaxPool1d(kernel_size=3))  # >> (batch_size, num_channels, (seq_len-6-18-18-18-18-18)/(3*3*3))

        conv_output_size = self.num_channels * ((self.seq_len - 96) // 27)
        
        fc_1 = nn.Sequential(nn.Linear(conv_output_size, self.linear_size), nn.ReLU(), nn.Dropout(self.dropout_rate))
        fc_2 = nn.Sequential(nn.Linear(self.linear_size, self.linear_size), nn.ReLU(), nn.Dropout(self.dropout_rate))
        fc_3 = nn.Sequential(nn.Linear(self.linear_size, self.output_size))
        
        self.conv_layers = nn.Sequential(conv_1, conv_2, conv_3, conv_4, conv_5, conv_6)
        self.fc_layers = nn.Sequential(fc_1, fc_2, fc_3)

        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, x):

        embedded_sequence = self.embeddings(x)
        embedded_sequence = embedded_sequence.permute(0, 2, 1)  # >> embedded_sequence: (batch_size, embedding_size, seq_len)

        conv_out = self.conv_layers(embedded_sequence)
        conv_out = conv_out.view(conv_out.shape[0], -1)

        logits = self.fc_layers(conv_out)
        preds = self.softmax(logits)

        return preds

