

import torch
from torch import nn



# 策略网络
class PolicyNet(nn.Module):

    def __init__(self, in_channels, channels_lst, action_dim) -> None:
        """
        Params:
            channels: list, 每一次卷积输出的通道数
            action_dim: 输出维度
        """
        super(PolicyNet, self).__init__()

        self.in_channels = in_channels
        self.layers = []

        for i in range(len(channels_lst)):
            self.layers.append(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channels_lst[i], kernel_size=3)
            )
            self.layers.append(
                nn.ReLU()
            )
            self.layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.in_channels = channels_lst[i]

        # 在卷积池化之后，输出的维度是(128, 11, 8)，先flatten, 后面再跟全连接
        self.layers.append(
            nn.Flatten()
        )

        # 加全连接
        self.layers.append(
            nn.Linear(64*11*8, 1024)
        )
        self.layers.append(
            nn.ReLU()
        )
        self.layers.append(
            nn.Dropout(0.5)
        )

        # 加输出层
        self.layers.append(
            nn.Linear(1024, action_dim)
        )

        self.model = nn.Sequential(*self.layers)


    def forward(self, x):

        return self.model(x)
            


# 价值网络
# 对于传统的AC算法，对actor网络(策略网络)的训练还是基于最大action-value的，也就是传统的DQN训练模式
# 而对于critic网络(价值网络)的训练不是基于state-value(输出维度为1)，而是action-vlaue(所以输出维度是action_dim)
class ValueNet(nn.Module):

    def __init__(self, in_channels, channels_lst, output_dim) -> None:
        super(ValueNet, self).__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim

        self.layers = []

        for i in range(len(channels_lst)):
            self.layers.append(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channels_lst[i], kernel_size=3)
            )
            self.layers.append(
                nn.ReLU()
            )
            self.layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.in_channels = channels_lst[i]

        # 在卷积池化之后，输出的维度是(128, 11, 8)，先flatten, 后面再跟全连接
        self.layers.append(
            nn.Flatten()
        )

        # 加全连接
        self.layers.append(
            nn.Linear(64*11*8, 1024)
        )
        self.layers.append(
            nn.ReLU()
        )
        self.layers.append(
            nn.Dropout(0.5)
        )

        # 加输出层     
        self.layers.append(
            nn.Linear(1024, self.output_dim)
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):

        return self.model(x)
            
