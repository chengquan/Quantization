import torch
import torch.nn as nn

from quant.quant_layers import QuantifiedLinear, QuantifiedConv2d


def _conv2d(in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            batch_norm=False,
            input_bit_width=8,
            weight_bit_width=8,
            bias_bit_width=8):
    conv = QuantifiedConv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias,
                            input_bit_width=input_bit_width,
                            weight_bit_width=weight_bit_width,
                            bias_bit_width=bias_bit_width
                            )
    if batch_norm:
        return nn.Sequential(conv,
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(conv,
                             nn.ReLU(inplace=True))


def _linear(in_features,
            out_features,
            bias=True,
            dropout=False,
            input_bit_width=8,
            weight_bit_width=8,
            bias_bit_width=8):
    linear = QuantifiedLinear(in_features=in_features,
                              out_features=out_features,
                              bias=bias,
                              input_bit_width=input_bit_width,
                              weight_bit_width=weight_bit_width,
                              bias_bit_width=bias_bit_width)
    if dropout:
        return nn.Sequential(linear,
                             nn.ReLU(inplace=True),
                             nn.Dropout())
    else:
        return nn.Sequential(linear,
                             nn.ReLU(inplace=True))


class LeNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            #nn.MaxPool2d(2, 2),
            #_conv2d(1, 1, 5, padding=0, input_bit_width=8, weight_bit_width=8, bias=False),
            #nn.MaxPool2d(2, 2),
            #_conv2d(1, 16, 5, padding=0, input_bit_width=8, weight_bit_width=8, bias=False),
            ##nn.MaxPool2d(2, 2)
            _conv2d(1, 6, 5, padding=0, input_bit_width=8, weight_bit_width=8, bias=False),
            nn.MaxPool2d(2, 2),
            _conv2d(6, 16, 5, padding=0, input_bit_width=8, weight_bit_width=8, bias=False),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            _linear(16*5*5, 120, bias=False, input_bit_width=8,
                    weight_bit_width=8, bias_bit_width=8),
            _linear(120, 84, bias=False, input_bit_width=8,
                    weight_bit_width=8, bias_bit_width=8),
            QuantifiedLinear(84, num_classes, bias=False, input_bit_width=8,
                             weight_bit_width=8, bias_bit_width=8)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # from torch import Tensor
    import numpy as np
    from torch.optim import SGD
    from torch.nn import CrossEntropyLoss
    device = torch.device('cpu')
    lenet = LeNet(10).to(device)

    ori_image = torch.from_numpy(np.random.randn(8, 1, 28, 28)).float().to(device)
    # np.random.rand
    ori_label = torch.from_numpy(np.random.randint(0, 10, (8,))).long().to(device)

    loss_fn = CrossEntropyLoss()
    # optim = Adam(vgg.parameters(), lr=1e-3)
    optim = SGD(lenet.parameters(), lr=1e-3)

    for i in range(10000):
        image = ori_image.clone()
        label = ori_label.clone()
        optim.zero_grad()
        out = lenet(image)
        # for p in vgg.parameters():
        #     if float(p.max().detach().cpu().numpy()) > 1e2:
        #         hook = 0
        loss = loss_fn(out, label)
        print(float(loss.cpu().detach().numpy()))
        loss.backward()
        nn.utils.clip_grad_norm_(vgg.parameters(), 5)
        optim.step()
    hook = 0
