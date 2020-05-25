
import torch

from models.deep_seq_net import DeepSeqNet
from torch import nn


class FastText(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, hidden_size, output_size, optimizer, learning_rate):

        super(FastText, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        self.fc_1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        embedded_sequence = self.embeddings(x)
        logits = self.fc_2(self.fc_1(embedded_sequence.mean(1)))
        return logits

