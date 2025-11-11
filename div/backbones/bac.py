import torch
import torch.nn as nn

from .shared import BackboneRegistry
from .bac_utils.norm import *
from .bac_utils.basic_unit import *


# @BackboneRegistry.register("bac")
class BAC(nn.Module):
   """Band-Aware Convolution Network for Diffusion, also called BCD in the paper."""

   @staticmethod
   def add_argparse_args(parser):
      parser.add_argument("--nblocks", type=int, required=False, default=8,
                          help="The number of Conv2Former blocks, 6 for small, 8 for mid and 16 for large.")
      parser.add_argument("--input_channel", type=int, required=False, default=4,
                          help="The number of input channels.")
      parser.add_argument("--hidden_channel", type=int, required=False, default=256,
                          help="The number of hidden channels, 32 for small, 256 for mid, and 384 for large.")
      parser.add_argument("--f_kernel_size", type=int, required=False, default=9,
                          help="Kernel size along the sub-band axis.")
      parser.add_argument("--t_kernel_size", type=int, required=False, default=11,
                          help="Kernel size along the frame axis.")
      parser.add_argument("--mlp_ratio", type=int, required=False, default=1,
                          help="MLP ratio for expansion.")
      parser.add_argument("--ada_rank", type=int, required=False, default=32,
                          help="Lora rank for ada-sola, 8 for small, 32 for mid, and 48 for large.")
      parser.add_argument("--ada_alpha", type=int, required=False, default=32,
                          help="Lora alpha for ada-sola, 8 for small, 32 for mid, and 48 for large.")
      parser.add_argument("--ada_mode", type=str, required=False, default="sola",
                          help="Ada LN mode.")
      parser.add_argument("--act_type", type=str, required=False, default="gelu",
                          help="Activation type.")
      parser.add_argument("--pe_type", type=str, required=False, default="positional",
                          choices=["positional", "gaussian"])
      parser.add_argument("--scale", type=int, required=False, default=1000,
                          help="1000 when timestep is (0,1) else 1 for ddxm family.")
      parser.add_argument("--decode_type", type=str, required=False, default="ri",
                          help="Spectrum decoding strategy.")
      parser.add_argument("--use_adanorm", action="store_false",
                          help="Whether to use AdaNorm strategy.")
      parser.add_argument("--causal", action="store_true",
                          help="Whether to use causal network setups.")
      return parser

   def __init__(self, 
                nblocks: int,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int = 1,
                ada_rank: int = 16,
                ada_alpha: int = 16,
                ada_mode: str = "none",
                act_type: str = "gelu",
                pe_type: str = "positional",
                scale: int = 1,
                decode_type: str = "ri",
                use_adanorm: bool = False,
                causal: bool = False,
                **unused_kwargs,
                ):
      super(BAC, self).__init__()
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
      self.pe_type = pe_type
      self.scale = scale
      self.decode_type = decode_type
      self.use_adanorm = use_adanorm
      self.causal = causal

      self.enc = SharedBandSplit_NB24_24k(input_channel=self.input_channel,
                                          feature_dim=self.hidden_channel,
                                          use_adanorm=self.use_adanorm,
                                          causal=self.causal,
                                          )
      self.nband = self.enc.get_nband()
      self.dec = SharedBandMerge_NB24_24k(nband=self.nband,
                                          feature_dim=self.hidden_channel,
                                          use_adanorm=self.use_adanorm,
                                          decode_type=self.decode_type)

      self.main_net = Conv2FormerNet(nband=self.nband,
                                    nblocks=self.nblocks,
                                    input_channel=self.hidden_channel,
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
      
      if self.pe_type == "positional":
         self.time_embed = PositionalTimestepEmbedder(hidden_size=self.hidden_channel, scale=self.scale)
      elif self.pe_type == "gaussian":
         self.time_embed = GaussianFourierProjection(embedding_size=self.hidden_channel)
         
      if self.ada_mode.lower() in ["vanilla", "single", "sola"] and self.use_adanorm:
         self.time_act = nn.SiLU()
         self.time_ada_final_nn1 = nn.Linear(self.hidden_channel, 2 * self.hidden_channel, bias=True)
         self.time_ada_final_nn2 = nn.Linear(self.hidden_channel, 2 * self.hidden_channel, bias=True)
         self.time_ada_begin_nn = nn.Linear(self.hidden_channel, 2 * self.hidden_channel, bias=True)
         if self.ada_mode.lower() in ["single", "sola"]:
            self.time_ada_nn = nn.Linear(self.hidden_channel, 6 * self.hidden_channel, bias=True)
         else:
            self.time_ada_nn = None

      self.alpha = nn.Parameter(1e-4 * torch.ones([1, self.hidden_channel, self.nband, 1]))

      self.initialize_weights()

   def initialize_weights(self):
      def _basic_init(module):
         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
               nn.init.constant_(module.bias, 0)
      self.apply(_basic_init)

      # Zero-out AdaLN
      if self.use_adanorm:
         self._init_ada()

   def _init_ada(self):
      if self.ada_mode.lower() in ["single", "sola"]:
         nn.init.constant_(self.time_ada_nn.weight, 0)
         nn.init.constant_(self.time_ada_nn.bias, 0)
      nn.init.constant_(self.time_ada_final_nn1.weight, 0)
      nn.init.constant_(self.time_ada_final_nn1.bias, 0)
      nn.init.constant_(self.time_ada_final_nn2.weight, 0)
      nn.init.constant_(self.time_ada_final_nn2.bias, 0)
      nn.init.constant_(self.time_ada_begin_nn.weight, 0)
      nn.init.constant_(self.time_ada_begin_nn.bias, 0)
      if self.ada_mode.lower() == "vanilla":
         for block in self.main_net.net:
            nn.init.constant_(block.ada.time_nn.weight, 0)
            nn.init.constant_(block.ada.time_nn.bias, 0)
      elif self.ada_mode.lower() == "sola":
         for block in self.main_net.net:
            nn.init.kaiming_uniform_(block.ada.lora_a.weight, a=math.sqrt(5))
            nn.init.constant_(block.ada.lora_b.weight, 0) 

   def forward(self, inpt, cond=None, time_cond=None):
      """
      inpt: (B, 2, F, T)
      cond: (B, 2, F, T)
      time_cond: (B,)
      return: (B, 2, F, T)
      """

      if time_cond.ndim < 1:
         time_cond = time_cond.unsqueeze(0)
      time_token = self.time_embed(time_cond)
      time_ada, time_ada_begin, time_ada_final1, time_ada_final2 = None, None, None, None
      if self.use_adanorm:
         time_token = self.time_act(time_token)
         if self.time_ada_nn is not None:
            time_ada = self.time_ada_nn(time_token)
         time_ada_final1 = self.time_ada_final_nn1(time_token)
         time_ada_final2 = self.time_ada_final_nn2(time_token)
         time_ada_begin = self.time_ada_begin_nn(time_token)

      inpt_spec = torch.cat([inpt, cond], dim=1)
      # band split
      enc_x = self.enc(inpt_spec, time_ada_begin=time_ada_begin)
      x = enc_x
      # sub-band modeling
      x = self.main_net(enc_x, time_token=time_token, time_ada=time_ada)

      x = x + self.alpha * enc_x

      # band merge, different reconstrcution strategies
      if self.decode_type.lower() == "mag+phase":
         cur_mag, cur_pha = self.dec(x, time_ada_final1=time_ada_final1, time_ada_final2=time_ada_final2)
         out_real, out_imag = cur_mag * torch.cos(cur_pha), cur_mag * torch.sin(cur_pha)
         out = torch.stack([out_real, out_imag], dim=1)
      elif self.decode_type.lower() == "ri":
         out_real, out_imag = self.dec(x, time_ada_final1=time_ada_final1, time_ada_final2=time_ada_final2)
         out = torch.cat([out_real, out_imag], dim=1)
      else:
         raise NotImplementedError("Only mag+phase and ri are supported, please check it carefully!")
      return out


if __name__ == "__main__":
   net = BAC(nblocks=8,
             input_channel=4,
             hidden_channel=256,
             f_kernel_size=5,
             t_kernel_size=11,
             mlp_ratio=1,
             ada_rank=32,
             ada_alpha=32,
             ada_mode="sola",
             act_type="gelu",
             pe_type="positional",
             scale=1,
             decode_type="ri",
             use_adanorm=True,     
             causal=False,        
             ).cuda()
   inpt = torch.randn([3, 2, 513, 64]).cuda()
   cond = torch.randn([3, 2, 513, 64]).cuda()
   time_cond = torch.rand([3]).cuda()
   out = net(inpt, cond, time_cond)
   print(f"inpt: {inpt.shape}->{out.shape}")
