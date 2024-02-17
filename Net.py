import torch
import torch.nn as nn
import math


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        # Input to hidden weights
        self.weight_xh = None

        # Hidden to hidden biases
        self.weight_hh = None

        # Input to hidden biases
        self.bias_xh = None

        # Hidden to hidden biases
        self.bias_hh = None

        # LSTM parameters
        self.weight_xh = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias_xh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        # Inizializziamo i parametri dell'LSTM

        std = 1.0 / math.sqrt(self.hidden_size)
        self.weight_xh.data.uniform_(-std, std)
        self.weight_hh.data.uniform_(-std, std)
        self.bias_xh.data.uniform_(-std, std)
        self.bias_hh.data.uniform_(-std, std)

    def forward(self, x):
        x = x.transpose(0, 1)

        # Unpack dimensions
        tt, n, h = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (n, h)
        h0 = torch.zeros(n, h, device=x.device)
        c0 = torch.zeros(n, h, device=x.device)

        # Define a list to store outputs. We will then stack them.
        y = []

        ht_1 = h0
        ct_1 = c0
        for t in range(tt):
            # LSTM update rule
            xh = torch.addmm(self.bias_xh, x[t], self.weight_xh)
            hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            it = torch.sigmoid(xh[:, 0:h] + hh[:, 0:h])
            ft = torch.sigmoid(xh[:, h:2 * h] + hh[:, h:2 * h])
            gt = torch.tanh(xh[:, 2 * h:3 * h] + hh[:, 2 * h:3 * h])
            ot = torch.sigmoid(xh[:, 3 * h:4 * h] + hh[:, 3 * h:4 * h])
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y.append(ht)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        # Stack the outputs. After this operation, output will have shape of
        # (tt, n, h)
        y = torch.stack(y)

        # Switch time and batch dimension, (tt, n, h) -> (n, tt, h)
        y = y.transpose(0, 1)
        return y


class BrakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            'cnn': nn.Sequential(
                # (3 x 227 x 227)
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=0),
                nn.ReLU(),
                # (96 x 111 x 111)
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(96),
                # (96 x 55 x 55)
                nn.Conv2d(in_channels=96, out_channels=384, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                # (384 x 26 x 26)
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(384),
                # (384 x 13 x 13)
                nn.Conv2d(in_channels=384, out_channels=512, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                # (512 x 13 x 13)
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                # (512 x 13 x 13)
                nn.Conv2d(in_channels=512, out_channels=384, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                # (384 x 13 x 13)
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=7, stride=2, padding=2),
                nn.ReLU(),
                # (384 x 6 x 6)
                nn.Flatten(),
                # (384 * 6 * 6 = 13824)
                nn.Linear(in_features=384*6*6, out_features=4096),
                # (4096)
            ),
            'lstm': nn.LSTM(input_size=4096, hidden_size=256),
            'out': nn.Sequential(
                nn.Linear(256, 2),
                nn.Softmax(dim=1)
            )
        })

    def forward(self, x):
        x = self.model['cnn'](x)

        x = x.view(x.shape[0], 1, -1)

        x = self.model['lstm'](x)[0][-1]

        y = self.model['out'](x)[0]

        return y

