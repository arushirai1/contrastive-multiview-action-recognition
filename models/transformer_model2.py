import torch
from torch import nn
from torch.autograd import Variable
import math, copy, time
import torch.nn.functional as F

### UTILS ###
def flatten(x):
    x = x.view(x.shape[0], x.shape[1], -1)
    return x

def unflatten(x, T, H, W):
    x = x.view(x.shape[0], -1, T, H, W)
    return x

### MODULES ###
class MultiHeadedAttention(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, num_heads=2, T=1, H=7, W=7):
        super(SelfAttention, self).__init__()
        self.T = T
        self.H = H
        self.W = W
        self.num_heads = num_heads
        self.key_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.val_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.upsample = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.scaling_param = nn.Parameter(torch.tensor(0.0001), requires_grad=True)

    def forward(self, x, query):
        key = flatten(self.key_conv(x)).transpose(-1,-2) # shape: (batch_size, d_model, THW)
        query = flatten(self.query_conv(query)) # shape: (batch_size, THW, d_model)
        value = flatten(self.val_conv(x)) # shape: (batch_size, THW, d_model)

        # scaled dot-product attention with respect with the heads
        attention = torch.matmul(key, query) / math.sqrt(key.shape[-1]) # shape: (batch_size, THW, THW)
        attention =  F.softmax(attention, -1)
        output = torch.matmul(attention, value.transpose(-1,-2))
        output = unflatten(output, self.T, self.H, self.W)
        output = self.upsample(output)
        x = torch.add(x, torch.mul(output, self.scaling_param)) # add back to original input with scaling
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, T=1, H=7, W=7):
        super(SelfAttention, self).__init__()
        self.T = T
        self.H = H
        self.W = W
        self.key_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.val_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.upsample = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.scaling_param = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, x, query):
        key = flatten(self.key_conv(x)).transpose(-1,-2) # shape: (batch_size, d_model, THW)
        query = flatten(self.query_conv(query)) # shape: (batch_size, THW, d_model)
        value = flatten(self.val_conv(x)) # shape: (batch_size, THW, d_model)

        # scaled dot-product attention
        attention = torch.matmul(key, query) / math.sqrt(key.shape[-1]) # shape: (batch_size, THW, THW)
        attention =  F.softmax(attention, -1)
        output = torch.matmul(attention, value.transpose(-1,-2))
        output = unflatten(output, self.T, self.H, self.W)
        output = self.upsample(output)
        x = torch.add(x, torch.mul(output, self.scaling_param)) # add back to original input with scaling
        return x

class TransformerModel(nn.Module):
    def __init__(self, base_model, num_classes, d_model=128, N=3, h=2, dropout=0.3, endpoint=''):
        super(TransformerModel, self).__init__()
        self.base_model = base_model
        self.endpoint = endpoint
        in_channels = 512*1*7*7

        if self.endpoint in ['layer4', 'layer3']:
            if endpoint == 'layer4':
                in_channels = 512
                T, H, W = (1, 7, 7)
            elif endpoint == 'layer3':
                in_channels = 256
                T, H, W = (4, 14, 14)

            self.attention = SelfAttention(in_channels=in_channels, out_channels=d_model, T=T, H=H, W=W)
            self.avg_pool = nn.AvgPool3d(kernel_size=(T,H,W))

        self.classifier = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))
        self.initialize_parameters()

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, x):
        x = self.base_model(x)

        if self.endpoint in ['layer4', 'layer3']:
            x = self.attention(x, x)
            x = self.avg_pool(x)
            x = x.squeeze()
            x = self.classifier(x)

        #x = self.classifier(x.view(x.shape[0], -1))
        return x