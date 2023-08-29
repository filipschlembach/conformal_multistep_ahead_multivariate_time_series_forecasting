import logging
import torch
import torch.utils.data


class LinearRegression(torch.nn.Module):
    def __init__(self, window_l: int = 1, horizon_l: int = 1,
                 window_f: int = 1, horizon_f: int | list = 1,
                 path=None, device: str = None, **kwargs):
        """
        Initialises the Linear Regression model.
        :param window_l: number of time steps in the input time-series
        :param horizon_l:  number of time steps in the forecast
        :param window_f: dimensionality of the input time-series
        :param horizon_f: dimensionality of the forecast
        :param path: optional path where to save the model
        :param device: cuda device. If none cpu is used
        :param kwargs: unused, for compatibility here
        """
        super(LinearRegression, self).__init__()
        if kwargs:
            for k, v in kwargs.items():
                logging.debug(f'Parameter {k} = {v} will be ignored.')

        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.wl = window_l
        self.hl = horizon_l
        self.wf = window_f
        self.hf = horizon_f
        self.input_size = self.wl * self.wf
        self.output_size = self.hl * self.hf
        self.path = path
        self.device = device

        # model layers
        self.linear = torch.nn.Linear(self.input_size, self.output_size)

        if self.device is not None:
            self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1).float()
        out = self.linear(x)
        return out.reshape(-1, self.hl, self.hf), None  # two return values to be compatible with the RNN implementation

    def fit(self, train_dataset: torch.utils.data.Dataset, batch_size: int, epochs: int, lr: float):
        """
        Trains the auxiliary forecaster to the training dataset.
        :param train_dataset: a dataset of type `torch.utils.data.Dataset`
        :param batch_size: batch size
        :param epochs: number of training epochs
        :param lr: learning rate
        :return: Mean train loss per epoch
        """
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        train_losses = []
        for epoch in range(epochs):
            train_loss = 0.0

            for x, y in train_loader:
                if self.device is not None:
                    x = x.to(self.device).float()
                    y = y.to(self.device).float()
                optimizer.zero_grad()

                out, _ = self(x)

                loss = criterion(out, y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            train_losses.append(mean_train_loss)
            if epoch % 99 == 0:
                logging.info("Epoch: {}\tTrain loss: {}".format(epoch, mean_train_loss))

        if self.path is not None:
            torch.save(self, self.path)

        return train_losses
