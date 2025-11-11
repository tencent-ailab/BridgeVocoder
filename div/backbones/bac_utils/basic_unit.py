import math
import numpy as np
import torch
import torch.nn as nn
from .norm import *


class GaussianFourierProjection(nn.Module):
   """
   Gaussian Fourier embeddings for noise levels.
   """
   def __init__(self, embedding_size=256, scale=16.0):
      super().__init__()
      self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
      self.mlp = nn.Sequential(
         nn.Linear(embedding_size * 2, embedding_size * 2, bias=True),
         nn.SiLU(),
         nn.Linear(embedding_size * 2, embedding_size, bias=True),
      )

   def forward(self, x):
      x_proj = torch.log(x[:, None]) * self.W[None, :] * 2 * np.pi  # 这里要取log
      x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
      x = self.mlp(x)
      return x


class PositionalTimestepEmbedder(nn.Module):
   """
   Embeds scalar timesteps into vector representations.
   """
   def __init__(self, 
                hidden_size, 
                frequency_embedding_size=256, 
                pe_type="positional",
                scale=1000,
                out_size=None):
      super().__init__()
      if out_size is None:
         out_size = hidden_size
      self.mlp = nn.Sequential(
         nn.Linear(frequency_embedding_size, hidden_size, bias=True),
         nn.SiLU(),
         nn.Linear(hidden_size, out_size, bias=True),
      )
      self.frequency_embedding_size = frequency_embedding_size
      self.scale = scale

   def forward(self, t):
      t_freq = timestep_embedding(t, self.frequency_embedding_size, scale=self.scale).type(
         self.mlp[0].weight.dtype)
      t_emb = self.mlp(t_freq)
      return t_emb


def timestep_embedding(timesteps, dim, max_period=10000, scale=1000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None] * scale  # 通过scale将timestep从[0, 1]放大
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Conv2FormerUnit(nn.Module):
   def __init__(self,
                nband: int,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int = 1,
                ada_rank: int = 8,
                ada_alpha: int = 8,
                ada_mode: str = "none",
                act_type: str = "gelu",
                causal: bool = False,
                use_adanorm: bool = False,
               ):
      super(Conv2FormerUnit, self).__init__()
      self.nband = nband
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.ada_rank = ada_rank
      self.ada_alpha = ada_alpha 
      self.ada_mode = ada_mode
      self.act_type = act_type
      self.causal = causal
      self.use_adanorm = use_adanorm
      #
      self.norm1 = BandwiseC2LayerNorm(self.nband, self.input_channel)
      self.act = self.set_act_layer()
      if self.use_adanorm:
         self.ada = AdaLN(self.input_channel,
                          self.ada_mode, 
                          self.ada_rank, 
                          self.ada_alpha,
                         )
      if self.causal:
         pad_ = nn.ConstantPad2d([t_kernel_size - 1, 0, f_kernel_size // 2, f_kernel_size // 2], value=0.)
      else:
         pad_ = nn.ConstantPad2d([t_kernel_size // 2, t_kernel_size // 2, f_kernel_size // 2, f_kernel_size // 2], value=0.)
      self.attn = nn.Sequential(
         nn.Conv2d(self.input_channel, self.hidden_channel, 1),
         self.act,
         pad_,
         nn.Conv2d(self.hidden_channel, self.hidden_channel, kernel_size=(self.f_kernel_size, self.t_kernel_size), groups=self.hidden_channel)
      )
      self.v = nn.Conv2d(self.input_channel, self.hidden_channel, 1)
      self.proj = nn.Conv2d(self.hidden_channel, self.input_channel, 1)
      
      # Feedforward
      self.norm2 = BandwiseC2LayerNorm(self.nband, self.input_channel)
      self.fc1 = nn.Sequential(
         nn.Conv2d(self.input_channel, self.input_channel * self.mlp_ratio, 1),
         self.act
      )
      if self.causal:
         pad_ = nn.ConstantPad2d([2, 0, 1, 1], value=0.)
      else:
         pad_ = nn.ConstantPad2d([1, 1, 1, 1], value=0.)
      self.dw_conv = nn.Sequential(
         pad_,
         nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, 3, groups=self.input_channel * self.mlp_ratio),
         self.act
      )
      self.fc2 = nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel, 1)

   def set_act_layer(self):
      if self.act_type.lower() == "relu":
         return nn.ReLU()
      elif self.act_type.lower() == "silu":
         return nn.SiLU()
      elif self.act_type.lower() == "gelu":
         return nn.GELU()
      elif self.act_type.lower() == "approgelu":
         return ApproximateGELU(dim=self.input_channel, dim_out=self.input_channel)

   def forward(self, x, time_token=None, time_ada=None):
      """
      x: (B, C, nband, T)
      time_token: (B, C)
      time_ada: (B, C) or None
      """
      if self.use_adanorm:
         time_ada = self.ada(time_token, time_ada)
         (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = time_ada.chunk(6, dim=1)
         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shift_msa.squeeze(1), \
                                                                          scale_msa.squeeze(1), \
                                                                          gate_msa.squeeze(1), \
                                                                          shift_mlp.squeeze(1), \
                                                                          scale_mlp.squeeze(1), \
                                                                          gate_mlp.squeeze(1)
      # conv-attention
      x_res = x
      if self.use_adanorm:
         x = film_modulate(self.norm1(x), shift_msa, scale_msa)
         x = self.attn(x) * self.v(x)
         x = self.proj(x)
         x = x_res + (1 - gate_msa[..., None, None]) * x
      else:
         x = self.norm1(x)
         x = self.attn(x) * self.v(x)
         x = self.proj(x)
         x = x_res + x
      # mlp
      x_res = x
      if self.use_adanorm:
         x = film_modulate(self.norm2(x), shift_mlp, scale_mlp)
         x = self.fc1(x)
         x = x + self.dw_conv(x)
         x = self.fc2(x)
         out = x_res + (1 - gate_mlp[..., None, None]) * x
      else:
         x = self.norm2(x)
         x = self.fc1(x)
         x = x + self.dw_conv(x)
         x = self.fc2(x)
         out = x_res + x
      
      return out


class Conv2FormerNet(nn.Module):   
   def __init__(self,
                nband: int,
                nblocks: int,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int = 1,
                ada_rank: int = 8,
                ada_alpha: int = 8,
                ada_mode: str = "none",
                act_type: str = "gelu",
                causal: bool = False,
                use_adanorm: bool = False,
                ):
      super(Conv2FormerNet, self).__init__()
      self.nband = nband
      self.nblocks = nblocks
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.ada_rank = ada_rank
      self.ada_alpha = ada_alpha 
      self.ada_mode = ada_mode
      self.act_type = act_type
      self.causal = causal
      self.use_adanorm = use_adanorm
      #
      net = []
      for _ in range(self.nblocks):
         net.append(
            Conv2FormerUnit(nband=self.nband,
                             input_channel=self.input_channel,
                             hidden_channel=self.hidden_channel,
                             f_kernel_size=self.f_kernel_size,
                             t_kernel_size=self.t_kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             ada_rank=self.ada_rank,
                             ada_alpha=self.ada_alpha,
                             ada_mode=self.ada_mode,
                             act_type=self.act_type,
                             causal=self.causal,
                             use_adanorm=self.use_adanorm,
                             )
         )
      
      self.net = nn.ModuleList(net)
   
   def forward(self, inpt, time_token=None, time_ada=None):
      """
      inpt: (B, C, nband, T)
      time_token: (B, C)
      time_ada: (B, C)
      """
      x = inpt
      # encoding
      for layer in self.net:
         x = layer(x, time_token=time_token, time_ada=time_ada)
      return x


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class SharedBandSplit_NB24_24k(nn.Module):
   def __init__(self,
                input_channel: int = 4,
                feature_dim: int = 64,
                use_adanorm: bool = False,
                causal: bool = True,
               ):
      super(SharedBandSplit_NB24_24k, self).__init__()
      self.input_channel = input_channel
      self.feature_dim = feature_dim
      self.use_adanorm = use_adanorm
      self.causal = causal
      self.eps = torch.finfo(torch.float32).eps
      
      if self.causal:
         pad = nn.ConstantPad2d([2, 0, 0, 0], value=0.)
      else:
         pad = nn.ConstantPad2d([1, 1, 0, 0], value=0.)
      
      self.reg1_encoder = nn.Sequential(
          pad,
          nn.Conv2d(in_channels=self.input_channel, out_channels=self.feature_dim, kernel_size=(12, 3), stride=(12, 1)),
          BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          pad,
          nn.Conv2d(in_channels=self.input_channel, out_channels=self.feature_dim, kernel_size=(24, 3), stride=(24, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          pad,
          nn.Conv2d(in_channels=self.input_channel, out_channels=self.feature_dim, kernel_size=(44, 3), stride=(44, 1)),
          BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim)
      )
      self.nband = 12 + 8 + 4
      print(f'Totally splitting {self.nband} bands for sampling rate: 24k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None, time_ada_begin=None):
      """
      input: (B, 4, F, T)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      batch_size = input.shape[0]
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      if self.use_adanorm:
         shift, scale = time_ada_begin.reshape(batch_size, 2, -1).chunk(2, dim=1)
         shift, scale = shift.squeeze(1), scale.squeeze(1)
         out = film_modulate(out, shift, scale)

      return out


class SharedBandMerge_NB24_24k(nn.Module):
   def __init__(self,
                nband: int,
                feature_dim: int = 64,
                use_adanorm: bool = False,
                decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB24_24k, self).__init__()
      self.nband = nband
      self.feature_dim = feature_dim
      self.use_adanorm = use_adanorm
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps

      self.norm1 = BandwiseC2LayerNorm(nband=self.nband, feature_dim=self.feature_dim)
      self.norm2 = BandwiseC2LayerNorm(nband=self.nband, feature_dim=self.feature_dim)
      if self.decode_type.lower() == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=2, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=2, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=2, kernel_size=(44, 1), stride=(44, 1))
         )
      elif self.decode_type.lower() == "ri":
         self.reg1_real_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_real_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_real_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim * 2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim * 2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
      else:
         raise NotImplementedError("Only Mag+Phase and RI are supported for decoding!")

   def forward(self, emb_input, time_ada_final1=None, time_ada_final2=None):
      """
      emb_input: (B, C, nband, T)
      time_ada_final1: (B, C) or None
      time_ada_final2: (B, C) or None
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      batch_size = emb_input.shape[0]
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)
      if self.use_adanorm:
         shift1, scale1 = time_ada_final1.reshape(batch_size, 2, -1).chunk(2, dim=1)
         shift1, scale1 = shift1.squeeze(1), scale1.squeeze(1)
         emb_input1 = film_modulate(emb_input1, shift1, scale1)
         shift2, scale2 = time_ada_final2.reshape(batch_size, 2, -1).chunk(2, dim=1)
         shift2, scale2 = shift2.squeeze(1), scale2.squeeze(1)
         emb_input2 = film_modulate(emb_input2, shift2, scale2)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :12].contiguous(), \
                         emb_input1[:, :, 12:20].contiguous(), \
                         emb_input1[:, :, 20:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :12].contiguous(), \
                         emb_input2[:, :, 12:20].contiguous(), \
                         emb_input2[:, :, 20:].contiguous()

      if self.decode_type.lower() == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_phase_decoder(x2_1), self.reg2_phase_decoder(x2_2), self.reg3_phase_decoder(x2_3)
         mag = torch.cat([mag1, mag2, mag3], dim=-2) 
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return torch.exp(mag.squeeze(1)), pha
      elif self.decode_type.lower() == "ri":
         real1, real2, real3 = self.reg1_real_decoder(x1_1), self.reg2_real_decoder(x1_2), self.reg3_real_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real, imag
