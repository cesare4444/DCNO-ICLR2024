import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchinfo import summary
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., activation='gelu'):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        elif activation == 'tanh':
            act = nn.Tanh
        else:
            raise NameError('invalid activation')

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=16):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.fourier_weight1 = nn.Parameter(
            torch.empty(in_channels, out_channels,
                        modes1, modes2, 2))
        self.fourier_weight2 = nn.Parameter(
            torch.empty(in_channels, out_channels,
                        modes1, modes2, 2))

        nn.init.xavier_uniform_(self.fourier_weight1, gain=1 / (in_channels * out_channels)
                                                           * np.sqrt((in_channels + out_channels) / init_scale))
        nn.init.xavier_uniform_(self.fourier_weight2, gain=1 / (in_channels * out_channels)
                                                           * np.sqrt((in_channels + out_channels) / init_scale))

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),  # a[..., 0], b[..., 0]都是实部，a[..., 1], b[..., 1]都是虚部
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):

        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight2)
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class DilatedConv(nn.Module):
    def __init__(self, num_channels=48, dilation=[1, 3, 9]):
        super(DilatedConv, self).__init__()

        self.convlist = nn.ModuleList([])
        self.depth = len(dilation)
        for i in range(self.depth):
            self.convlist.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3,
                                           dilation=dilation[i],
                                           padding=dilation[i]))

    def forward(self, x):
        for i in range(self.depth-1):
            x = self.convlist[i](x)
            # x = F.relu(x)
            x = F.gelu(x)
        x = self.convlist[-1](x)
        return x
# --------------------------------------------------------------------------------
class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm, stride=2, patch_padding=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)

        return x

class Process(nn.Module):
    def __init__(self, modes=12, width=64, num_spectral_layers=5, dilation=[1, 3, 9], padding=5, add_pos=False):
        super().__init__()

        self.modes = modes
        self.add_pos = add_pos
        if add_pos:
            self.width = width + 2
        else:
            self.width = width
        self.num_spectral_layers = num_spectral_layers
        self.padding = padding  # pad the domain if input is non-periodic

        self.Flayers_a = nn.ModuleList([])
        self.Flayers_b = nn.ModuleList([])
        self.Clayers = nn.ModuleList([])
        for _ in range(num_spectral_layers):
            self.Flayers_a.append(SpectralConv2d_fast(self.width, self.width, self.modes, self.modes))
            self.Flayers_b.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1))
            self.Clayers.append(DilatedConv(self.width, dilation))

    def forward(self, x, pos=None):
        if self.add_pos:
            x = torch.cat((x, pos), dim=-1)

        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.num_spectral_layers):
            x1 = self.Flayers_a[i](x)
            x2 = self.Flayers_b[i](x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = x
            x2 = self.Clayers[i](x)
            x = x1 + x2

        return x

class Decoder(nn.Module):
    def __init__(self, modes=12, width=64, hidden_dim=128,  padding=5):
        super().__init__()

        self.modes = modes
        self.width = width
        self.padding = padding  # pad the domain if input is non-periodic

        self.Flayer1_a = SpectralConv2d_fast(self.width, self.width, self.modes, self.modes)
        self.Flayer1_b = nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Flayer2_a = SpectralConv2d_fast(self.width, self.width, self.modes, self.modes)
        self.Flayer2_b = nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1)

        self.FFD = FeedForward(self.width, hidden_dim, 1)

    def forward(self, x):
        x1 = self.Flayer1_a(x)
        x2 = self.Flayer1_b(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.Flayer2_a(x)
        x2 = self.Flayer2_b(x)
        x = x1 + x2

        if self.padding:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.FFD(x)

        return x

class DCNO2d(nn.Module):
    def __init__(self, R_dic):
        super(DCNO2d, self).__init__()
        self.feature_dim = R_dic['feature_dim']
        self.dropout = 0.0
        self.resout = R_dic['res_output']
        self.FNO_padding = R_dic['FNO_padding']

        self.addpos = R_dic['addpos'] if 'addpos' in R_dic else 0

        self.encoder = PatchEmbed(patch_size=R_dic['patch_size'], in_chans=R_dic['in_dim'],
                                  embed_dim=self.feature_dim, stride=R_dic['subsample_stride'],
                                  patch_padding=R_dic['patch_padding'])

        self.process = Process(modes=R_dic['modes'], width=R_dic['feature_dim'],
                               num_spectral_layers=R_dic['num_spectral_layers'],
                               dilation=R_dic['dilation'], padding=R_dic['FNO_padding'])

        self.decoder = Decoder(modes=R_dic['modes'], width=R_dic['feature_dim'], hidden_dim=R_dic['mlp_hidden_dim'], padding=R_dic['FNO_padding'])

        if 'y_norm' in R_dic:
            self.y_norm = R_dic['y_norm']
        else:
            self.y_norm = None

    def forward(self, x):
        if self.addpos:
            self.grid = self.grid.to(x.device)
            x = torch.cat((x, self.grid), dim=-1)
        # x.shape = [b, 1, res_input,res_input]

        x = self.encoder(x)
        B, L, C = x.shape
        x = x.view(B, self.resout, self.resout, C)
        x = x.permute(0, 3, 1, 2)
        # x.shape = [b, feature, res_out, res_out]

        x = self.process(x)
        # x.shape = [b,feature,res_out,res_out]

        x = self.decoder(x)
        # x.shape = [b,res_out,res_out,1]

        if self.y_norm is not None:
            x = torch.squeeze(x, dim=-1)
            x = self.y_norm.decode(x)
            x = torch.unsqueeze(x, dim=-1)

        return torch.squeeze(x)

# =========================================================================================

if __name__ == "__main__":
    R_dic = {}
    R_dic['subsample_stride'] = 1
    R_dic['patch_padding'] = 1
    R_dic['patch_size'] = 3
    R_dic['res_input'] = 256
    R_dic['res_output'] = 256

    R_dic['dilation'] = [1, 3, 9]
    R_dic['feature_dim'] = 32 # feature dim, in order to enhance expressiveness
    R_dic['modes'] = 12  # modes of FNO
    R_dic['num_spectral_layers'] = 3
    R_dic['mlp_hidden_dim'] = 128
    R_dic['in_dim'] = 1
    R_dic['FNO_padding'] = 5
    model = DCNO2d(R_dic)
    summary(model, input_size=(4, 1, 256, 256))