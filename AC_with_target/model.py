

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

        self.model = nn.Sequential(*self.layers)



    def forward(self, x):

        return self.model(x)
            

