import torch.nn as nn
import torch


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res

class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class SelfAttention(nn.Module):
    def __init__(self, f_bins):
        super(SelfAttention, self).__init__()
        self.linear = nn.Linear(f_bins, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, T, F = x.shape   # [batch, channels, time, freq]
        x_mean_t = x.mean(dim=2)   # [B, C, F]
        fcn = self.linear(x_mean_t)  # (B, in_ch)
        alpha = self.sigmoid(fcn)   # (B, in_ch)
        
        alpha = alpha.view(B, C, 1, 1)  # [B, C, 1, 1]
        return alpha

class FrequencyPositionalEncoding(nn.Module):
    """
    학습 가능한 주파수 위치 임베딩: shape = [1, 1, 1, F]
    """
    def __init__(self, f_bins: int):
        super().__init__()
        self.f_bins = f_bins
        self.p_freq = nn.Parameter(torch.zeros(f_bins))  #bin

    def forward(self, x):  # x: [B, C, T, F]
        B, C, T, F = x.shape
        pe = self.p_freq.view(1, 1, 1, F)
        pe = pe.expand(B,C,T,F)
        return pe
    
class fac_conv(nn.Module):
    def __init__(self, f_bins):
        super(fac_conv, self).__init__()
        self.self_attention = SelfAttention(f_bins)
        self.pe = FrequencyPositionalEncoding(f_bins)
        # self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    def forward(self, x):
        attn_out = self.self_attention(x)  # must match x shape
        pe = self.pe(x)
        x = x + attn_out * pe
        # x = self.conv2d(x)
        return x
    
class Q_fac_conv(nn.Module):
    def __init__(self, f_bins):
        super(Q_fac_conv, self).__init__()
        self.self_attention = SelfAttention(f_bins)
        self.pe = FrequencyPositionalEncoding(f_bins)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, x):
        x = self.dequant(x)
        attn_out = self.self_attention(x)
        pe = self.pe(x)
        x = x + attn_out * pe
        x = self.quant(x)
        return x


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="cg",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        fac_layers=[1, 1, 1, 1, 1, 1, 1], 
        freq_bins=[128, 64, 32, 16, 8, 4, 2, 1],
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]

            if i == 0:
                cnn.add_module(f"fac_conv{i}", Q_fac_conv(f_bins=freq_bins[i])),
                cnn.add_module(f"conv{i}", 
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),)
            else:
                cnn.add_module(f"pre conv{i}", nn.Conv2d(nIn, nOut//2, 1, groups=2))
                cnn.add_module(f"fac_conv{i}", Q_fac_conv(f_bins=freq_bins[i],)),
                cnn.add_module(f"depthwiseconv{i}", nn.Conv2d(nOut//2, nOut//2, kernel_size[i], stride[i], padding[i], groups=nOut//2),)
                cnn.add_module(f"ChannelShuffle{i}", ChannelShuffle(2))
                cnn.add_module(f"point-wise conv{i}", nn.Conv2d(nOut//2, nOut, 1, groups=2))

            if activ.lower() == "relu":
                cnn.add_module(f"relu{i}", nn.ReLU(inplace=True))
            elif activ.lower() == "cg":
                cnn.add_module(f"cg{i}", ContextGating(nOut))
            cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels
        
        cnn.add_module(f"adaptivepool", nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.cnn(x)
        return x


class shuffleFAC(nn.Module):
        def __init__(self,
                    n_input_ch,
                    n_class=10,
                    activation="glu",
                    conv_dropout=0.5,
                    nb_filters=[64, 64, 64],
                    **convkwargs):
            super().__init__()
            
            self.cnn = CNN(n_in_channel=n_input_ch, activation=activation, conv_dropout=conv_dropout, nb_filters=nb_filters, **convkwargs)

            self.fc = nn.Linear(nb_filters[-1], 5)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
            )
        def forward(self, x, return_feats=False):
            x = x.transpose(2,3)
            x = self.cnn(x)
            fmap = x
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            if return_feats:
                return x, fmap
            else:
                return x