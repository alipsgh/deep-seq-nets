
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn


class TextCNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, num_channels, kernel_size, max_seq_len, output_size,
                 dropout_rate, optimizer, learning_rate):

        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.max_seq_len = max_seq_len
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        self.conv_1 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels, kernel_size=self.kernel_size[0]),
                                    nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels, kernel_size=self.kernel_size[1]),
                                    nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels, kernel_size=self.kernel_size[2]),
                                    nn.ReLU())
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.num_channels * len(self.kernel_size), self.output_size)

        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, x):

        embedded_sequence = self.embeddings(x)
        embedded_sequence = embedded_sequence.permute(0, 2, 1)
        
        feature_map_1 = torch.max(self.conv_1(embedded_sequence), dim=2)[0]
        feature_map_2 = torch.max(self.conv_2(embedded_sequence), dim=2)[0]
        feature_map_3 = torch.max(self.conv_3(embedded_sequence), dim=2)[0]

        feature_vector = torch.cat((feature_map_1, feature_map_2, feature_map_3), 1)
        feature_vector = self.dropout(feature_vector)

        logits = self.fc(feature_vector)
        preds = self.softmax(logits)

        return preds

