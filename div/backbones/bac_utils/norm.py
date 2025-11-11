import torch
import torch.nn as nn
from torch.nn import Parameter, init


def film_modulate(x, shift, scale):
   return x * (1 + scale[..., None, None]) + shift[..., None, None]


class AdaLN(nn.Module):
   def __init__(self, 
               dim: int, 
               ada_mode: str = "ada",
               ada_rank: int = 32,
               ada_alpha: int = 32,
               ):
      super(AdaLN, self).__init__()
      self.dim = dim
      self.ada_mode = ada_mode
      self.ada_rank = ada_rank
      self.ada_alpha = ada_alpha
      self.scale_shift_table = None

      if self.ada_mode.lower() == "vanilla":
         self.time_nn = nn.Linear(dim, 6 * dim, bias=True)
      elif self.ada_mode.lower() == "single":
         self.scale_shift_table = nn.Parameter(torch.zeros(6, dim), requires_grad=True)
      elif self.ada_mode.lower() == "sola":
         self.lora_a = nn.Linear(dim, ada_rank * 6, bias=False)
         self.lora_b = nn.Linear(ada_rank * 6, dim * 6, bias=False)
         self.scaling = self.ada_alpha / self.ada_rank
      else:
         raise NotImplementedError("only vanilla, single, and sola are provided in AdaLN, please check it carefully!")

   def forward(self, time_token=None, time_ada=None):
      batch_size = time_token.shape[0]
      if self.ada_mode == "vanilla":
         assert time_ada is None, "for vanilla, time_ada should not be provided!"
         time_ada = self.time_nn(time_token).reshape(batch_size, 6, -1)
      elif self.ada_mode.lower() == "single": 
         time_ada = time_ada.reshape(batch_size, 6, -1)
         time_ada = self.scale_shift_table[None] + time_ada
      elif self.ada_mode.lower() == "sola":
         time_ada_lora = self.lora_b(self.lora_a(time_token)) * self.scaling
         time_ada = time_ada + time_ada_lora
         time_ada = time_ada.reshape(batch_size, 6, -1)
      else:
         raise NotImplementedError("only none/vanilla/single/sola modes are supported for AdaLN!")

      return time_ada


class ChannelNormalization(nn.Module):
   def __init__(self, num_channels, ndim=3, affine=True):
      super(ChannelNormalization, self).__init__()
      self.num_channels = num_channels
      self.ndim = ndim
      self.affine = affine
      self.eps = 1e-5
      if affine:
         if ndim == 3:
               self.gain = Parameter(torch.empty([1, num_channels, 1]))
               self.bias = Parameter(torch.empty([1, num_channels, 1]))
         elif ndim == 4:
               self.gain = Parameter(torch.empty([1, num_channels, 1, 1]))
               self.bias = Parameter(torch.empty([1, num_channels, 1, 1]))
      else:
         self.register_parameter('gain', None)
         self.register_parameter('bias', None)
      # 
      self.reset_parameters()

   def reset_parameters(self):
      if self.gain is not None and self.bias is not None:
         init.constant_(self.gain, 1.)
         init.constant_(self.bias, 0.)

   def forward(self, input):
      """
      input: (B, C, T) or (B, C, X, T)
      return: xxx
      """
      if input.ndim == 3:
         mean_ = input.mean(dim=1, keepdims=True)
         std_ = torch.sqrt(torch.var(input, dim=1, keepdims=True, unbiased=False) + self.eps)
      elif input.ndim == 4:
         mean_ = input.mean(dim=1, keepdims=True)
         std_ = torch.sqrt(torch.var(input, dim=1, keepdims=True, unbiased=False) + self.eps)
      x = (input - mean_) / std_

      if self.affine:
         x = x * self.gain + self.bias

      return x


class BandwiseC2LayerNorm(nn.Module):
   def __init__(self,
               nband: int,
               feature_dim: int,
               affine = True,
               ):
      super(BandwiseC2LayerNorm, self).__init__()
      self.nband = nband
      self.feature_dim = feature_dim
      self.affine = affine
      self.eps = 1e-5
      self.gain_matrix = Parameter(torch.ones([1, feature_dim, nband, 1]))
      self.bias_matrix = Parameter(torch.zeros([1, feature_dim, nband, 1]))

   def forward(self, input):
      """
      input: (B, C, nband, T)
      return: (B, C, nband, T)
      """
      mean_ = torch.mean(input, dim=1, keepdim=True)  # (B, 1, nband, T)
      std_ = torch.sqrt(torch.var(input, dim=1, unbiased=False, keepdim=True) + self.eps)  # (B, 1, nband, T)

      if self.affine:
         output = self.gain_matrix * ((input - mean_) / std_) + self.bias_matrix 
      else:
         output = (input - mean_) / std_
      
      return output