import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CNN(nn.Module):
    def __init__(self, out_dim, image_dims, learning_params):

        super(CNN, self).__init__()
        self.image_dims = image_dims
        self.learning_params = learning_params

        # image_size = (256, 256)
        # input_channels = 1, output_channels = 16
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # image_size = (128, 128)
        # input channels = 16, output_channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # image_size = (64, 64)
        # input channels = 32, output_channels = 32
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # image_size = (32, 32)
        # input channels = 32, output_channels = 32
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # image_size = (16, 16)
        # input_neurons = 32*16*16, output_neurons = 1024
        if list(self.image_dims) == [64, 64]:
            self.fc1 = nn.Linear(32 * 4 * 4, 1024)  # 64
        if list(self.image_dims) == [128, 128]:
            self.fc1 = nn.Linear(32 * 8 * 8, 1024)  # 128
        if list(self.image_dims) == [256, 256]:
            self.fc1 = nn.Linear(32 * 16 * 16, 1024)  # 256

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_dim)

    def forward(self, x):
        if self.learning_params['apply_batchnorm']:
            x = self.pool1(F.elu(self.conv1_bn(self.conv1(x))))
            x = self.pool2(F.elu(self.conv2_bn(self.conv2(x))))
            x = self.pool3(F.elu(self.conv3_bn(self.conv3(x))))
            x = self.pool4(F.elu(self.conv4_bn(self.conv4(x))))
        else:
            x = self.pool1(F.elu(self.conv1(x)))
            x = self.pool2(F.elu(self.conv2(x)))
            x = self.pool3(F.elu(self.conv3(x)))
            x = self.pool4(F.elu(self.conv4(x)))

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.learning_params['dropout'])
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.learning_params['dropout'])
        x = self.fc3(x)
        return x
