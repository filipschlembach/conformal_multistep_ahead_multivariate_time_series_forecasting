# Adapted from Kamilė Stankevičiūtė https://github.com/kamilest/conformal-rnn
import logging

import torch
import torch.utils.data


class PointRNN(torch.nn.Module):
    """
    The auxiliary RNN issuing point predictions.

    Point predictions are used as baseline to which the (normalised)
    uncertainty intervals are added in the main CFRNN network.
    """

    def __init__(self, window_f: int = 1, horizon_f: int = 1, horizon_l: int = 1, rnn_mode: str = "LSTM",
                 embedding_size: int = 10, path=None, device: str = None, **kwargs):
        """
        Initialises the RNN model.
        :param window_f: input_size = |wf|, dimensionality of the input time-series
        :param horizon_f: output_size = |hf|, dimensionality of the forecast
        :param horizon_l: horizon = |hl|, forecasting horizon
        :param rnn_mode: type of the underlying RNN cell, select between 'RNN', 'LSTM' and 'GRU'
        :param embedding_size: size of the latent RNN embeddings
        :param path: optional path where to save the model
        :param device: cuda device. If none cpu is used.
        """

        super(PointRNN, self).__init__()
        if kwargs:
            for k, v in kwargs.items():
                logging.debug(f'Parameter {k} = {v} will be ignored.')
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series
        self.input_size = window_f
        self.embedding_size = embedding_size
        self.output_size = horizon_f
        self.horizon = horizon_l
        self.path = path
        self.device = device

        # rnn layer(s)
        self.rnn_mode = rnn_mode
        if self.rnn_mode == "RNN":
            self._rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=embedding_size, batch_first=True)
        elif self.rnn_mode == "GRU":
            self._rnn = torch.nn.GRU(input_size=self.input_size, hidden_size=embedding_size, batch_first=True)
        elif self.rnn_mode == 'LSTM':
            self._rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=embedding_size, batch_first=True)
        else:
            raise ValueError('Unsupported RNN cell type.')
        # fully connected layer
        self._out = torch.nn.Linear(embedding_size, self.horizon * self.output_size)

        if self.device is not None:
            self.to(self.device)

    def forward(self, x: torch.Tensor, state=None):
        if state is not None:
            h_0, c_0 = state
        else:
            h_0 = None

        if self.device is not None:
            x = x.to(self.device)

        # todo: change so that it can deal with wf != hf
        # [batch, horizon, output_size]
        if self.rnn_mode == "LSTM":
            _, (h_n, c_n) = self._rnn(x.float(), state)
        else:
            _, h_n = self._rnn(x.float(), h_0)
            c_n = None

        out = self._out(h_n).reshape(-1, self.horizon, self.output_size)

        return out, (h_n, c_n)

    def fit(self, train_dataset: torch.utils.data.Dataset, batch_size: int, epochs: int, lr: float) -> None:
        """
        Trains the auxiliary forecaster to the training dataset.
        :param train_dataset: a dataset of type `torch.utils.data.Dataset`
        :param batch_size: batch size
        :param epochs: number of training epochs
        :param lr: learning rate
        :return: None
        """
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0

            for x, y in train_loader:
                if self.device is not None:
                    x = x.to(self.device)
                    y = y.to(self.device)
                optimizer.zero_grad()

                out, _ = self(x)

                lengths = [x.shape[1]]
                mask = PointRNN.get_lengths_mask(x, lengths, self.horizon)
                if self.device is not None:
                    mask = mask.to(self.device)
                valid_out = out * mask

                loss = criterion(valid_out.float(), y.float())
                loss.backward()

                train_loss += loss.item()  # todo: figure out why the training loss doesn't relly change despite the model's performance increasing.

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 50 == 0:
                logging.info("Epoch: {}\tTrain loss: {}".format(epoch, mean_train_loss))

        if self.path is not None:
            torch.save(self, self.path)

    @staticmethod
    def get_lengths_mask(sequences, lengths, horizon):
        """
        Returns the mask indicating which positions in the sequence are valid.

        Args:
            sequences: (batch of) input sequences
            lengths: the lengths of every sequence in the batch
            horizon: the forecasting horizon
        """

        lengths_mask = torch.zeros(sequences.size(0), horizon, sequences.size(2))
        for i, l in enumerate(lengths):
            lengths_mask[i, : min(l, horizon), :] = 1

        return lengths_mask
