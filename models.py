import torch.nn as nn


class LargeLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.nlayers = 15
        self.input_size = 4
        self.hidden_size = 200

        self.lstm1 = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.nlayers,
                             batch_first=True,
                             )

        self.lstm2 = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.nlayers,
                             batch_first=True,
                             )

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size,
                               out_channels=self.hidden_size,
                               kernel_size=9,
                               dilation=1,
                               padding=4,
                               stride=1
                               )

        self.conv2 = nn.Conv1d(in_channels=self.hidden_size,
                               out_channels=self.hidden_size,
                               kernel_size=5,
                               dilation=1,
                               padding=2,
                               stride=1
                               )

        self.linear1 = nn.Linear(self.hidden_size, 100)
        self.linear2 = nn.Linear(100, 200)
        self.linear3 = nn.Linear(200, 4)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, (h, c) = self.lstm1(x)
        x = self.activation(x)
        x = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, (_, _) = self.lstm2(x, (h, c))
        x = self.activation(x)
        x = self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.activation(x)
        x = self.linear1(x)

        x = self.activation(x)
        x = self.linear2(x)

        x = self.activation(x)
        x = self.linear3(x)

        return x


class SmallLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.nlayers = 2
        self.input_size = 4
        self.hidden_size = 10

        self.lstm1 = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.nlayers,
                             batch_first=True,
                             )

        self.lstm2 = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.nlayers,
                             batch_first=True,
                             )

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size,
                               out_channels=self.hidden_size,
                               kernel_size=9,
                               dilation=1,
                               padding=4,
                               stride=1
                               )

        self.conv2 = nn.Conv1d(in_channels=self.hidden_size,
                               out_channels=self.hidden_size,
                               kernel_size=5,
                               dilation=1,
                               padding=2,
                               stride=1
                               )

        self.linear1 = nn.Linear(self.hidden_size, 20)
        self.linear2 = nn.Linear(20, 30)
        self.linear3 = nn.Linear(30, 4)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, (h, c) = self.lstm1(x)
        x = self.activation(x)
        x = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, (_, _) = self.lstm2(x, (h, c))
        x = self.activation(x)
        x = self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.activation(x)
        x = self.linear1(x)

        x = self.activation(x)
        x = self.linear2(x)

        x = self.activation(x)
        x = self.linear3(x)

        return x
