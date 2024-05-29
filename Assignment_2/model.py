import torch
from torch import nn
import torch.nn.functional as F
        
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
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(16 * 8 * 3, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
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
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
        - x: input tensor
        
        Returns:
        - x: output tensor
        """
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 8 * 3)
        x = self.dropout1(self.bn3(F.relu(self.fc1(x))))
        x = self.dropout2(self.bn4(F.relu(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()