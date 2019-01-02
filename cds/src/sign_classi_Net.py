import torch as tr, torch.nn as nn, torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() #3x64x64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5) #60 (in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        #pool 30
        self.conv2 = nn.Conv2d(32, 64,3) #28
        #pool 14
        self.conv3 = nn.Conv2d(64, 128, 3) #12
        #pool 6
        self.conv4 = nn.Conv2d(128, 256, 3) #4
        #pool 2
        self.fc1 = nn.Linear(256 *2 *2, 2048, True) #
        self.fc2 = nn.Linear(2048, 1024, True)
        self.fc3 = nn.Linear(1024, 3, True)


    def forward(self, X): #3*32*32 #3*227*227
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = F.relu(self.conv3(X))
        X = self.pool(X)
        X = F.relu(self.conv4(X))
        X = self.pool(X)

        X = X.view(X.size(0), -1)

        X = tr.tanh(self.fc1(X))
        X = tr.tanh(self.fc2(X))
        X = F.softmax(self.fc3(X), dim=1)
        return X