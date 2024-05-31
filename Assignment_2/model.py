import torch
import torch.nn.functional as F
from torch import nn


class SimpleCNN(nn.Module):
    """
    Simple CNN model for binary classification

    Args:
    - dropout_rate: dropout rate

    Returns:
    - output: predicted probability of the input belonging to the positive class
    """

    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(16 * 8 * 3, 120)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(84, 1)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the model

        Returns:
        - None
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        """
        Forward pass of the model

        Args:
        - x: input tensor

        Returns:
        - x: output tensor
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 3)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()
