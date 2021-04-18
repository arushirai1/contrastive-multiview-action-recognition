import torch
from torch import nn
from torch.autograd import Variable
import math, copy, time
import torch.nn.functional as F
from einops import rearrange, repeat
from math import log, pi
### TESTING INSTRUCTIONS
'''
3) add the positional embedding in the transformer module
'''

### UTILS ###
def flatten(x):
    x = x.view(x.shape[0], x.shape[1], -1)
    return x

def unflatten(x, T, H, W):
    x = x.view(x.shape[0], -1, T, H, W)
    return x

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    "From https://github.com/lucidrains/perceiver-pytorch"
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

### MODULES ###
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        b, _, *axis = x.shape

        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device = x.device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(pos, 10, 4, base=2)
        enc_pos = rearrange(enc_pos, '... n d -> (n d) ... ')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)

        x = torch.cat((x, enc_pos), dim=1)
        return x

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).repeat_interleave(x.shape[1], dim=1)
        std = x.std(dim=1, keepdim=True).repeat_interleave(x.shape[1], dim=1)

        return (self.a_2 * (x - mean).permute(0,2,3,4,1) / (std + self.eps).permute(0,2,3,4,1) + self.b_2).permute(0,4,1,2,3)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv3d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.w_2 = nn.Conv3d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, num_heads=2, T=1, H=7, W=7, dropout=0.3):
        super(MultiHeadedAttention, self).__init__()
        assert out_channels % num_heads == 0
        self.d_k = out_channels // num_heads
        self.dropout = nn.Dropout(p=dropout)

        self.num_heads = num_heads

        self.key_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.val_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.upsample = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x, query):
        T, H, W = x.shape[2:5]
        key = flatten(self.key_conv(x)) # shape: (batch_size, THW, d_model)
        query = flatten(self.query_conv(query)) # shape: (batch_size, THW, d_model)
        value = flatten(self.val_conv(x)) # shape: (batch_size, THW, d_model)

        # reshape with respect to the heads
        key, query, value = [projection.view(x.shape[0], -1, self.num_heads, self.d_k).transpose(1,2) for projection in (key, query, value)]

        # scaled dot-product attention with respect with the heads
        attention = torch.matmul(key.transpose(-2, -1), query) / math.sqrt(key.shape[-1]) # shape: (batch_size, heads, THW, THW)
        attention = F.softmax(attention, -1)
        output = torch.matmul(attention, value.transpose(-1,-2))

        # "Concat" the outputs of each head using a view and apply then apply another 1x1 conv
        output = output.transpose(1, 2).contiguous() \
            .view(output.shape[0], -1, self.num_heads * self.d_k)

        output = unflatten(output, T, H, W)
        output = self.upsample(output)

        return output

class SelfAttention(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(SelfAttention, self).__init__()
        self.key_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.val_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.upsample = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.scaling_param = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, x, query):
        T, H, W = x.shape[2:5]
        key = flatten(self.key_conv(x)).transpose(-1,-2) # shape: (batch_size, d_model, THW)
        query = flatten(self.query_conv(query)) # shape: (batch_size, THW, d_model)
        value = flatten(self.val_conv(x)) # shape: (batch_size, THW, d_model)

        # scaled dot-product attention
        attention = torch.matmul(key, query) / math.sqrt(key.shape[-1]) # shape: (batch_size, THW, THW)
        attention =  F.softmax(attention, -1)
        output = torch.matmul(attention, value.transpose(-1,-2))
        output = unflatten(output, T, H, W)
        output = self.upsample(output)
        x = torch.add(x, torch.mul(output, self.scaling_param)) # add back to original input with scaling
        return x

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, T=1, H=7, W=7, dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, query):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, query))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, query):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, query)
        return self.norm(x)

class TransformerModel(nn.Module):
    def __init__(self, base_model, num_classes, d_model=128, N=3, h=2, dropout=0.3, endpoint='', positional_flag=False, eval_mode=False):
        super(TransformerModel, self).__init__()
        self.base_model = base_model
        self.endpoint = endpoint
        in_channels = 512*1*7*7
        c = copy.deepcopy

        if positional_flag:
            positional_embedding_size = 27
        else:
            positional_embedding_size = 0

        if self.endpoint in ['layer4', 'layer3', 'avgpool']:
            if endpoint == 'layer4':
                in_channels = 512 + positional_embedding_size
            elif endpoint == 'layer3':
                in_channels = 256 + positional_embedding_size
            elif endpoint == 'avgpool':
                # use temporal attention module
                in_channels = 512 + positional_embedding_size
            if positional_flag:
                self.positional_embedding = PositionalEncoding(in_channels, dropout)
            else:
                self.positional_embedding = None

            feed_forward = PositionwiseFeedForward(in_channels, in_channels * 2, dropout)
            attention = MultiHeadedAttention(in_channels=in_channels, out_channels=d_model, num_heads=h)

            self.encoder = Encoder(EncoderLayer(in_channels, c(attention), feed_forward=c(feed_forward), dropout=dropout), N)
            #self.avg_pool = nn.AvgPool3d(kernel_size=(T,H,W))
            self.avg_pool = lambda input: nn.functional.adaptive_avg_pool3d(input, 1)

        self.classifier = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))
        self.initialize_parameters()

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.base_model(x)
        if self.endpoint in ['layer4', 'layer3', 'avgpool']:
            if len(x.shape) < 5:
                x = rearrange(x, 'b clips channels -> b channels clips 1 1')
            if self.positional_embedding:
                x = self.positional_embedding(x)
            x = self.encoder(x, x)
            x = self.avg_pool(x)

            x = rearrange(x, 'b f 1 1 1 -> b f')

            x = self.classifier(x)

        return x


