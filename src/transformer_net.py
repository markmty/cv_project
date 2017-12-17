import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, X):
        hidden = F.relu(self.norm1(self.conv1(X)))
        res = X + self.norm2(self.conv2(hidden))
        return res


class transformer_net(nn.Module):
    def __init__(self):
        super(transformer_net, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(128)

        # residual layers
        self.res1 = Residual(128)
        self.res2 = Residual(128)
        self.res3 = Residual(128)
        self.res4 = Residual(128)
        self.res5 = Residual(128)

        # deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.norm5 = nn.BatchNorm2d(32)
        self.deconv3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, X):
        y = F.relu(self.norm1(self.conv1(X)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = F.relu(self.norm4(self.deconv1(y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        y = F.tanh(self.deconv3(y))

        return y
