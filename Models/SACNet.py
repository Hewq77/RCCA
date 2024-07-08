import torch
from torch import nn

class SACNet(nn.Module):
    def __init__(self, num_features=103, num_classes=9, conv_features=64, trans_features=32, K=48, D=32):
        super(SACNet, self).__init__()

        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,  # 3
                               bias=True)

        self.alpha3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.beta3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.gamma3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.deta3 = nn.Conv2d(trans_features, conv_features, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding=0,
                                  bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D, conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_cls = nn.Conv2d(conv_features * 3, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D

        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))
        conv1 = x

        x = self.relu(self.conv1(x))
        conv2 = x
        x = self.avgpool(x)

        x = self.relu(self.conv2(x))
        n, c, h, w = x.size()
        interpolation_context3 = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x_half = self.avgpool(x)
        n, c, h, w = x_half.size()
        alpha_x = self.alpha3(x_half)
        beta_x = self.beta3(x_half)
        gamma_x = self.relu(self.gamma3(x_half))

        alpha_x = alpha_x.squeeze().permute(1, 2, 0)
        # h*w x c
        alpha_x = alpha_x.view(-1, self.trans_features)
        # c x h*w
        beta_x = beta_x.view(self.trans_features, -1)
        gamma_x = gamma_x.view(self.trans_features, -1)

        context_x = torch.matmul(alpha_x, beta_x)
        context_x = F.softmax(context_x)

        context_x = torch.matmul(gamma_x, context_x)
        context_x = context_x.view(n, self.trans_features, h, w)
        context_x = interpolation_context3(context_x)

        deta_x = self.relu(self.deta3(context_x))
        x = deta_x + x

        Z = self.relu(self.encoding(x)).view(1, self.D, -1).permute(0, 2, 1)  # n,h*w,D

        A = F.softmax(scaled_l2(Z, self.codewords, self.scale), dim=2)  # b,n,k
        E = aggregate(A, Z, self.codewords)  # b,k,d
        E_sum = torch.sum(self.relu(self.BN(E)), 1)  # b,d
        gamma = self.sigmoid(self.attention(E_sum))  # b,num_conv
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma
        context3 = interpolation(x)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1, conv2, context3), 1)
        x = self.conv_cls(x)

        return x