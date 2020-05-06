
import numpy as np
import torch

from torch.nn import Module
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.optim.lr_scheduler import MultiplicativeLR


class DeepSeqNet(Module):

    def __init__(self):
        super(DeepSeqNet, self).__init__()

    def _compile(self, optimizer, learning_rate):
        self._set_optim(optimizer, learning_rate)
        self._set_scheduler()
        self._set_criterion()

    def _set_optim(self, optimizer, learning_rate):

        optimizer = optimizer.lower()
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def _set_scheduler(self):
        self.scheduler = MultiplicativeLR(self.optimizer, lr_lambda=(lambda x: 0.95))

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        raise NotImplementedError()

    def fit(self, x, y):

        self.train()

        self.optimizer.zero_grad()

        y_ = self.forward(x)

        loss = self.criterion(y_, y)
        loss.backward()

        self.optimizer.step()

        return loss

    def evaluate(self, data_iterator):

        self.eval()

        labels, preds = [], []
        for _, batch in enumerate(data_iterator):

            x = batch.text.t()
            if torch.cuda.is_available():
                x = x.cuda()

            y_ = self.forward(x)
            pred = torch.argmax(y_, 1)

            preds.extend(pred.cpu().numpy())
            labels.extend(batch.label.numpy())

        score = accuracy_score(labels, np.array(preds).flatten())

        return score

    def run_epoch(self, train_iterator, val_iterator):

        train_losses = []
        val_accuracies = []
        losses = []
        for i, batch in enumerate(train_iterator):

            x = batch.text.t()
            y = batch.label.type(torch.LongTensor)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            loss = self.fit(x, y)
            losses.append(loss.item())

            if i % 100 == 0 and i != 0:
                avg_train_loss = float(np.mean(losses))
                train_losses.append(avg_train_loss)
                losses = []

                val_accuracy = self.evaluate(val_iterator)
                print("Iteration: %4d | train loss: %3.2f | val acc.: %.2f" % ((i + 1), avg_train_loss * 100, val_accuracy * 100))

        # Run the scheduler to reduce the learning rate
        self.scheduler.step(epoch=None)

        return train_losses, val_accuracies

