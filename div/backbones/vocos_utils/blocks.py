from typing import Optional, Tuple
import math
import torch
from torch import nn, einsum
from torch.nn import init
from einops import rearrange
from torch.nn.utils import weight_norm, remove_weight_norm


class CondConvNeXtBlock(nn.Module):
    def __init__(
        self,
        sr_list: int,
        dim: int,
        intermediate_dim: int,
        ntokens: int,
        num_heads: int,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super(CondConvNeXtBlock, self).__init__()
        self.sr_list = sr_list
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.ntokens = ntokens
        self.num_heads = num_heads
        self.adanorm_num_embeddings = adanorm_num_embeddings

        self.cond_attn = ModulateAttention(in_channels=dim,
                                           hid_channels=dim,
                                           num_heads=num_heads,
                                           )
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        self.register_parameter('kernel_16k', nn.Parameter(torch.empty([self.ntokens, self.dim])))
        self.register_parameter('kernel_22k', nn.Parameter(torch.empty([self.ntokens, self.dim])))
        self.register_parameter('kernel_24k', nn.Parameter(torch.empty([self.ntokens, self.dim])))
        self.register_parameter('kernel_32k', nn.Parameter(torch.empty([self.ntokens, self.dim])))
        self.register_parameter('kernel_44k', nn.Parameter(torch.empty([self.ntokens, self.dim])))
        self.register_parameter('kernel_48k', nn.Parameter(torch.empty([self.ntokens, self.dim])))
        # 
        self.reset_params()
    
    def reset_params(self):
        init.kaiming_uniform_(self.kernel_16k, a=math.sqrt(5))
        init.kaiming_uniform_(self.kernel_22k, a=math.sqrt(5))
        init.kaiming_uniform_(self.kernel_24k, a=math.sqrt(5))
        init.kaiming_uniform_(self.kernel_32k, a=math.sqrt(5))
        init.kaiming_uniform_(self.kernel_44k, a=math.sqrt(5))
        init.kaiming_uniform_(self.kernel_48k, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None, sr_symbol: Optional[str] = None) -> torch.Tensor:
        """
        x: (B, C, T)
        aux: (L, C)
        """
        if sr_symbol == '16k':
            aux = self.kernel_16k
        elif sr_symbol == '22k':
            aux = self.kernel_22k
        elif sr_symbol == '24k':
            aux = self.kernel_24k
        elif sr_symbol == '32k':
            aux = self.kernel_32k
        elif sr_symbol == '44k':
            aux = self.kernel_44k
        elif sr_symbol == '48k':
            aux = self.kernel_48k

        x = self.cond_attn(x, aux)

        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextBlock2(nn.Module):
    def __init__(self, 
                 dim: int,
                 intermediate_dim: int,
                 layer_scale_init_value = None,
                 adanorm_num_embeddings = None,
                 ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv = nn.Linear(intermediate_dim, dim)
    
    def forward(self, x, cond_embedding_id = None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv(x)
        
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class ModulateAttention(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hid_channels: int,
                 num_heads: int,
                 ):
        super(ModulateAttention, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_heads = num_heads
        self.scale = (hid_channels / num_heads) ** (-0.25)

        self.q_linear = nn.Linear(in_channels, hid_channels)
        self.kv_linear = nn.Linear(in_channels, 2 * hid_channels)
        self.out_linear = nn.Linear(hid_channels, in_channels)
        #
        self._reset_parameters()
    
    def _reset_parameters(self):
        init.xavier_uniform_(self.q_linear.weight)
        init.xavier_uniform_(self.kv_linear.weight)
        init.xavier_uniform_(self.out_linear.weight)
        init.constant_(self.q_linear.bias, 0.)
        init.constant_(self.kv_linear.bias, 0.)
        init.constant_(self.out_linear.bias, 0.)

    def forward(self, inpt, aux):
        """
        inpt: (B, C, T)
        aux: (L, C)
        return: (B, C, T)
        """
        resi = inpt.clone()
        q = self.q_linear(inpt.transpose(-2, -1)) * self.scale
        k, v = self.kv_linear(aux).chunk(2, dim=-1)
        k = k * self.scale
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k, v = map(lambda t: rearrange(t, 't (h d) -> t h d', h=self.num_heads), (k, v))
        e = einsum('b i h d, j h d -> b i j h', q, k)
        attn = e.softmax(dim=-2).to(q.dtype)
        out = einsum('b i j h, j h d -> b i h d', attn, v)
        out = self.out_linear(out.flatten(-2))

        return resi + out.transpose(-2, -1)


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)