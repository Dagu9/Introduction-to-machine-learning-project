import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        
        # (256,256) => (252,252)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5))

        # max pool 2 (252,252) => (126,126)

        # (126,126) => (124,124)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3,3))
        
        # max pool 2 (124,124) => (62,62)

        # (62,62) => (60,60)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))

        #max pool 2 (60,60) => (30,30)

        # (30,30) => (28,28)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))

        # max pool 2 (28,28) => (14,14)

        # (14,14) => (12,12)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))

        #max pool 2 (12,12) => (6,6)

        self.fc1 = nn.Linear(in_features=6*6*128, out_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=512)

        self.fc3 = nn.Linear(in_features=512, out_features=256)

        self.fc4 = nn.Linear(in_features=256, out_features=128)

        self.fc5 = nn.Linear(in_features=128, out_features=64)

        self.fc6 = nn.Linear(in_features=64, out_features=num_classes)

        
    
    def forward(self, x):

        x = x.view(-1, 1, 256, 256)
        #size (B, 1, 256, 256)

        x = self.conv1(x)
        x = F.relu(x)     
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        #flatten data
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)

        return x
        