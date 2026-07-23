import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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
    def __init__(self, f_bins, in_ch):
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
    def __init__(self, f_bins, in_ch, out_ch, kernel_size, stride, padding):
        super(fac_conv, self).__init__()
        self.self_attention = SelfAttention(f_bins, in_ch)
        self.pe = FrequencyPositionalEncoding(f_bins)
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    def forward(self, x):
        attn_out = self.self_attention(x)  # must match x shape
        pe = self.pe(x)
        x = x + attn_out * pe
        x = self.conv2d(x)
        return x


class CNN(nn.Module):
    def __init__(
        self,
        n_input_ch,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        f_bins=[128, 64, 32, 16, 8, 4, 2, 1],
        fac_layers=[1, 1, 1, 1, 1, 1, 1], 
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s

        Args:
            n_input_ch: int, number of input channel
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
            nIn = n_input_ch if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]

            if fac_layers[i]==1:
                cnn.add_module(f"fac_conv{i}", fac_conv(f_bins[i], nIn, nOut, kernel_size[i], stride[i], padding[i]))  # unique name
            else:
                cnn.add_module(f"conv{i}", 
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
                )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels

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
    
    def get_features(self, x, kind='map'):
        """
            지식증류 시 feature map 추출
            map : [B C T F]
            frame : [B T C]
            vec : [B C]
        """
        fmap = self.cnn(x)
        if kind == 'map':
            return fmap
        elif kind == 'frame':
            B, C, T, F = fmap.shape
            if F != 1:
                return fmap.permute(0,2,1,3).contiguous().view(B, T, C*F)
            else:
                return fmap.squeeze(-1).permute(0, 2, 1)
        elif kind == 'vec':
            return torch.flatten(self.gap2d(fmap), 1)
        else:
            raise ValueError("kind must be one of {'map','frame','vec'}")

class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=256,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 specaugm_t_p=0.2,
                 specaugm_t_l=5,
                 specaugm_f_p=0.2,
                 specaugm_f_l=10,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.specaugm_t_p = specaugm_t_p
        self.specaugm_t_l = specaugm_t_l
        self.specaugm_f_p = specaugm_f_p
        self.specaugm_f_l = specaugm_f_l

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell, n_class)

        feat_dim = self.cnn.nb_filters[-1]
        self.feat_dim = feat_dim
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension


    def forward(self, x, return_feats : bool = False, feat_kind : str = 'map'): 
        x = x.transpose(2,3)
        x = self.cnn(x)
        fmap = x
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frame, ch*freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # x size : [bs, frames, chan]

        x = self.dropout(x)

        #classifier
        strong = self.dense(x) #strong size : [bs, frames, n_class]
        if self.attention:
            sof = self.dense_softmax(x) #sof size : [bs, frames, n_class]
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        if return_feats == True:
            return weak, fmap
        else:
            return weak

