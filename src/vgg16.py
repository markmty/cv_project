import torch.nn as nn
import torch.nn.functional as F

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        y = F.relu(self.conv1(X))
        y = F.relu(self.conv2(y))
        temp1 = y
        y = F.max_pool2d(y, kernel_size=2, stride=2)

        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        temp2 = y
        y = F.max_pool2d(y, kernel_size=2, stride=2)

        y = F.relu(self.conv5(y))
        y = F.relu(self.conv6(y))
        y = F.relu(self.conv7(y))
        temp3 = y
        y = F.max_pool2d(y, kernel_size=2, stride=2)

        y = F.relu(self.conv8(y))
        y = F.relu(self.conv9(y))
        y = F.relu(self.conv10(y))
        temp4 = y
        #y = F.max_pool2d(y, kernel_size=2, stride=2)

        return [temp1, temp2, temp3, temp4]