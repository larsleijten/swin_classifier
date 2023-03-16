import torch
import torch.nn as nn

class MLPhead(nn.Module):
    def __init__(self, input_shape):
        # Check if I need to adapt this?
        super(MLPhead, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_shape, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        #x = self.sigmoid(x)
        return x
    

class normMLPhead(nn.Module):
    def __init__(self, input_shape):
        super(MLPhead, self).__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.norm1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.norm1(self.fc1(x))
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.norm2(self.fc2(x))
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class oldMLPhead(nn.Module):
    def __init__(self, input_shape):
        super(oldMLPhead, self).__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x