import os
import numpy as np
import torch
import torch.nn as nn
from pytorch_model_summary import summary
from vit_pytorch.vit import ViT


def create_model(in_dim, out_dim, model_params, saved_model_dir=None, device='cpu'):

    if model_params['model_type'] == 'simple_cnn':
        model = CNN(
            in_dim=in_dim,
            in_channels=1,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] == 'nature_cnn':
        model = NatureCNN(
            in_dim=in_dim,
            in_channels=1,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] == 'resnet':
        model = ResNet(
            ResidualBlock,
            out_dim=out_dim,
            **model_params['model_kwargs'],
        ).to(device)

    elif model_params['model_type'] == 'vit':
        model = ViT(
            image_size=in_dim[0],
            channels=1,
            num_classes=out_dim,
            pool='mean',
            **model_params['model_kwargs']
        ).to(device)
    else:
        raise ValueError('Incorrect model_type specified:  %s' % (model_params['model_type'],))

    if saved_model_dir is not None:
        model.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_model.pth'), map_location='cpu')
        )

    print(summary(
        model,
        torch.zeros((1, 1, *in_dim)).to(device),
        show_input=True
    ))

    return model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CNN(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        conv_layers=[16, 16, 16],
        conv_kernel_sizes=[5, 5, 5],
        fc_layers=[128, 128],
        apply_batchnorm=False,
        dropout=0.0
    ):
        super(CNN, self).__init__()

        assert len(conv_layers) > 0, "conv_layers must contain values"
        assert len(fc_layers) > 0, "fc_layers must contain values"
        assert len(conv_layers) == len(conv_kernel_sizes), "conv_layers must contain same number of values as conv_kernel_sizes"

        # add first layer to network
        cnn_modules = []
        cnn_modules.append(nn.Conv2d(in_channels, conv_layers[0], kernel_size=conv_kernel_sizes[0], stride=1, padding=2))
        if apply_batchnorm:
            cnn_modules.append(nn.BatchNorm2d(conv_layers[0]))
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # add the remaining conv layers by iterating through params
        for idx in range(len(conv_layers) - 1):
            cnn_modules.append(
                nn.Conv2d(
                    conv_layers[idx],
                    conv_layers[idx + 1],
                    kernel_size=conv_kernel_sizes[idx + 1],
                    stride=1, padding=2)
                )

            if apply_batchnorm:
                cnn_modules.append(nn.BatchNorm2d(conv_layers[idx+1]))

            cnn_modules.append(nn.ReLU())
            cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # create cnn component of network
        self.cnn = nn.Sequential(*cnn_modules)

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, 1, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper (Commonly used in RL):
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    """

    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        fc_layers=[128, 128],
        dropout=0.0
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, 1, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, out_dim):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
