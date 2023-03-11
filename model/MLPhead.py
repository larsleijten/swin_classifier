import torch
import torch.nn as nn

class MLPhead(nn.Module):
    def __init__(self, input_shape):
        super(MLPhead, self).__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x