from math import ceil
import warnings
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import time

from collections import OrderedDict
from div import sampling
from div.sdes import SDERegistry
from div.backbones import BackboneRegistry
from div.util.inference import *
from div.util.other import pad_spec
from div.util.loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from div.backbones.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
import itertools
torch.autograd.set_detect_anomaly(True)

class ScoreModelGAN(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--opt_type", type=str, choices=['Adam', 'AdamW'], required=False, default='AdamW',
                            help='The optimizer type.')
        parser.add_argument("--lr", type=float, required=False, default=5e-4, 
                            help="The learning rate (5e-4 by default)")
        parser.add_argument("--beta1", type=float, default=0.8,
                            help="Beta1 for Adam/AdamW.")
        parser.add_argument("--beta2", type=float, default=0.99,
                            help="Beta2 for Adam/AdamW.")
        parser.add_argument("--ema_decay", type=float, default=0.999, 
                            help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03,
                            help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20,
                            help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type_list", type=str, required=False, default="score_mse:1.0",
                            help="The type of loss functions to use.")
        return parser

    def __init__(self,
        backbone: str = "blade",
        sde: str = "ouvesde",
        opt_type: str = 'AdamW',
        beta1: float = 0.9,
        beta2: float = 0.99,
        lr: float = 1e-4, 
        ema_decay: float = 0.999,
        t_eps: float = 3e-2, 
        nolog: bool = False,
        num_eval_files: int = 20, 
        loss_type_list = "score_mse:1.0", 
        data_module_cls = None, 
        max_epochs = 3100,
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a bridge-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.sde_name = sde

        if self.sde_name.lower() == "bridgegan":
            kwargs.update(scale_by_sigma=False)
            self.mpd = MultiPeriodDiscriminator().to(self.device)
            self.mrd = MultiResolutionDiscriminator().to(self.device)
            self.optim_d = torch.optim.AdamW(itertools.chain(self.mrd.parameters(), self.mpd.parameters()), lr=0.0001, betas=[0.8, 0.99])
            last_epoch = -1
            # if existing
            # state_dict_do = torch.load('./do_bridgeganGANbase', map_location = self.device)
            # self.mpd.load_state_dict(state_dict_do["mpd"])
            # self.mrd.load_state_dict(state_dict_do["mrd"])
            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999, last_epoch=last_epoch)

        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        kwargs.update(input_channels=4)  # cat(xt, y)
        self.dnn = dnn_cls(**kwargs)
        self.kwargs = kwargs
        self.max_epochs = max_epochs

        # Store hyperparams and save them
        self.opt_type = opt_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps

        if type(loss_type_list) is str:loss_type_list = loss_type_list.strip().split(',')
        self.loss_type_list = loss_type_list
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

        self._reduce_op_3 = lambda *args, **kwargs: torch.mean(torch.sum(dim=[1, 2], *args, **kwargs))
        self._reduce_op_4 = lambda *args, **kwargs: torch.mean(torch.sum(dim=[1, 2, 3], *args, **kwargs))

        self.nolog = nolog
        self.configure_losses()

    def configure_losses(self):
        # Denoiser loss
        self.loss_dict = {}
        self.weight_dict = {}
        for cur_loss_zip in self.loss_type_list:
            cur_loss, cur_weight = cur_loss_zip.split(":")
            cur_weight = float(cur_weight)
            self.weight_dict[cur_loss.lower()] = cur_weight
            if cur_loss.lower() == "multi-mel":
                self.loss_dict[cur_loss.lower()] = MultiresolutionMelLoss(sampling_rate=self.data_module.sampling_rate)
            elif cur_loss.lower() == "mel":
                self.loss_dict[cur_loss.lower()] = MelLoss(sampling_rate=self.data_module.sampling_rate)
            elif cur_loss.lower() == "score_mse":
                self.loss_dict[cur_loss.lower()] = lambda x: self._reduce_op_4(torch.square(torch.abs(x)))
            elif cur_loss.lower() == "score_mae":
                self.loss_dict[cur_loss.lower()] = lambda x: self._reduce_op_4(torch.abs(x))
            else:
                raise NotImplementedError

    def configure_optimizers(self):
        if self.opt_type == "Adam":
            optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif self.opt_type == "AdamW":
            optimizer = torch.optim.AdamW(self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # set scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, score_wav=None, x_wav=None, score_mag=None, x_mag=None, score_pha=None, x_pha=None):
        """
        err: (B, 2, F, T)
        score_wav: (B, L)
        x_wav: (B, L)
        score_mag: (B, F, T)
        x_mag: (B, F, T)
        score_pha: (B, F, T)
        x_pha: (B, F, T)
        """
        loss_val_dict = {}
        loss = 0.
        for k in self.loss_dict:
            if k.lower() in ["multi-mel", "mel"]:
                #print(x_wav, score_wav)
                cur_loss = self.loss_dict[k](x_wav, score_wav, self._reduce_op_3)
                loss = loss + self.weight_dict[k] * cur_loss
            elif k.lower() == "stft":
                cur_loss = self.loss_dict[k](x_mag, score_mag, self._reduce_op_3)
                loss = loss + self.weight_dict[k] * cur_loss
            elif k.lower() == "phase":
                cur_loss = self.loss_dict[k](x_pha, score_pha, x_pha.shape[1], x_pha.shape[-1], self._reduce_op_3)
                loss = loss + self.weight_dict[k] * cur_loss
            elif k.lower() == "score_mse":
                cur_loss = self._reduce_op_4(torch.square(torch.abs(err)))
                loss = loss + self.weight_dict[k] * cur_loss
            elif k.lower() == "score_mae":
                cur_loss = self._reduce_op_4(torch.abs(err))
                loss = loss + self.weight_dict[k] * cur_loss
            else:
                raise NotImplementedError
            loss_val_dict[k] = cur_loss.item()

        return loss, loss_val_dict

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def forward(self, x, t, y, **kwargs):
        """
        x: (B, 2, F, T)
        y: (B, 2, F, T)
        t: (B,)
        """
        score = self.dnn(x, cond=y, time_cond=t)

        return score

    def _mag_spec_fwd(self, spec):
        """
        spec: (B, F, T)
        return: (B, F, T)
        """
        if self.data_module.transform_type == "exponent":
            if self.data_module.spec_abs_exponent != 1:
                spec = (spec ** self.data_module.spec_abs_exponent) * self.data_module.spec_factor
        elif self.data_module.transform_type == "log":
            spec = torch.log(1 + spec)
        elif self.data_module.transform_type == "none":
            spec = spec
        return spec

    def _mag_spec_back(self, spec):
        """
        spec: (B, F, T)
        return: (B, F, T)
        """
        if self.data_module.transform_type == "exponent":
            if self.data_module.spec_abs_exponent != 1:
                spec = (spec / self.data_module.spec_factor) ** (1 / self.data_module.spec_abs_exponent)
        elif self.data_module.transform_type == "log":
            spec = torch.exp(spec) - 1
        elif self.data_module.transform_type == "none":
            spec = spec
        return spec

    def _step(self, batch, batch_idx):
        """
        x: (B, 1, F-1, T), target
        y: (B, 1, F-1, T), inpt
        """
        x, y, x_audio = batch
        x = torch.cat([x.real, x.imag], dim=1)  # (B, 2, F, T)
        y = torch.cat([y.real, y.imag], dim=1)  # (B, 2, F, T)
        x_audio = x_audio.squeeze(1)  # (B, L)
        real_len = x_audio.shape[-1]

        if self.sde_name.lower() == "ouve":
            # t: (B,)
            t = torch.rand(x.shape[0], device=x.device, requires_grad=False) * (self.sde.T - self.t_eps) + self.t_eps
            xt, target, std = self.sde.forward_diffusion(x0=x, x1=y, t=t)
            score = self(xt, t, y)
            err = score * std + target

            score_wav, score_mag, x_mag, score_pha, x_pha = None, None, None, None, None

        elif self.sde_name == "bridgegan":
                # discriminator update
            if batch_idx % 3==0:
                t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False)
                t = torch.clamp(t, self.sde.offset, 1.0 - self.sde.offset)
                xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)
                score = self(xt, t, y)# generator
                score_mag, score_pha = self._mag_spec_back(torch.norm(score, dim=1)), torch.atan2(score[:, -1], score[:, 0])
                x_mag = self._mag_spec_back(torch.norm(x, dim=1))
                x_pha = torch.atan2(x[:, -1], x[:, 0])
                if self.data_module.drop_last_freq:
                    last_score_freq_mag, last_score_freq_pha = score_mag[:, -1, None], score_pha[:, -1, None]
                    score_mag_ = torch.cat([score_mag, last_score_freq_mag], dim=1)
                    score_pha_ = torch.cat([score_pha, last_score_freq_pha], dim=1)

                score_decom = torch.complex(score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_))
                score_wav = torch.istft(score_decom,
                                        n_fft=self.data_module.n_fft,
                                        hop_length=self.data_module.hop_size,
                                        win_length=self.data_module.win_size,
                                        center=True,
                                        length=real_len,
                                        )# score_wav is x_esti_audio
                self.optim_d.zero_grad()
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(x_audio, score_wav)# T domain
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = DiscriminatorLoss(y_df_hat_r, y_df_hat_g)
                y_ds_hat_r, y_ds_hat_g, _, _ = self.mrd(x_audio, score_wav)
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = DiscriminatorLoss(y_ds_hat_r, y_ds_hat_g)
                L_D = (loss_disc_s * 1.0 + loss_disc_f)
                L_D.backward()
                self.optim_d.step()
            
            t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False)
            t = torch.clamp(t, self.sde.offset, 1.0 - self.sde.offset)
            xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)
            score = self(xt, t, y) # generator

            # score error
            err = score - target
            # other loss
            if len(self.loss_dict) > 1:
                score_mag, score_pha = self._mag_spec_back(torch.norm(score, dim=1)), torch.atan2(score[:, -1], score[:, 0])
                x_mag = self._mag_spec_back(torch.norm(x, dim=1))
                x_pha = torch.atan2(x[:, -1], x[:, 0])
                if self.data_module.drop_last_freq:
                    last_score_freq_mag, last_score_freq_pha = score_mag[:, -1, None], score_pha[:, -1, None]
                    score_mag_ = torch.cat([score_mag, last_score_freq_mag], dim=1)
                    score_pha_ = torch.cat([score_pha, last_score_freq_pha], dim=1)

                score_decom = torch.complex(score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_))
                score_wav = torch.istft(score_decom,
                                        n_fft=self.data_module.n_fft,
                                        hop_length=self.data_module.hop_size,
                                        win_length=self.data_module.win_size,
                                        center=True,
                                        length=real_len,
                                        )# score_wav is x_esti_audio

                _, y_df_g, fmap_f_r, fmap_f_g = self.mpd(x_audio, score_wav)
                _, y_ds_g, fmap_s_r, fmap_s_g = self.mrd(x_audio, score_wav)
                loss_fm_f = FeatureMatchingLoss(fmap_f_r, fmap_f_g)
                loss_fm_s = FeatureMatchingLoss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = GeneratorLoss(y_df_g)
                loss_gen_s, losses_gen_s = GeneratorLoss(y_ds_g)
                L_GAN_G = loss_gen_s * 1.0 + loss_gen_f
                L_FM = loss_fm_s * 1.0 + loss_fm_f


            loss, loss_val_dict = self._loss(err, score_wav=score_wav, x_wav=x_audio, score_mag=score_mag, x_mag=x_mag, score_pha=score_pha, x_pha=x_pha)
            loss1 = loss + 20.0 * (L_GAN_G + L_FM)
            return loss1, loss_val_dict#.detach_().requires_grad_(True)

        loss, loss_val_dict = self._loss(err, score_wav=score_wav, x_wav=x_audio, score_mag=score_mag, x_mag=x_mag, score_pha=score_pha, x_pha=x_pha)

        return loss, loss_val_dict

    def training_step(self, batch, batch_idx, **kwargs):
        loss, loss_val_dict = self._step(batch, batch_idx)
        for k in loss_val_dict:
            self.log(f'train_loss_{k}', loss_val_dict[k], on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss, loss_val_dict = self._step(batch, 1)  # batch_idx=1不更新L_D.backward()
        for k in loss_val_dict:
            self.log(f'valid_loss_{k}', loss_val_dict[k], on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, estoi_est, periodicity_est = evaluate_score_model(self, self.num_eval_files)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"Periodicity at epoch {self.current_epoch} : {periodicity_est:.2f}")
            print('__________________________________________________________________')
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)
            self.log('ValidationPeriodicity', periodicity_est, on_step=False, on_epoch=True)
        # save do_################
        if self.sde_name == "bridgegan":
            self.scheduler_d.step()
        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, scale_factor=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, sig):
        x = self._inv_mel(self._mel(sig))
        return x

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)
    
    def _mel(self, sig):
        return self.data_module.sig2mel(sig)
    
    def _inv_mel(self, mel):
        return self.data_module.inv_mel(mel)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def load_score_model(self, checkpoint):
        from collections import OrderedDict
        import io
        if isinstance(checkpoint, dict):
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            state_dict = torch.load(buffer, map_location=self.device)
        else:
            state_dict = torch.load(checkpoint, map_location=self.device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            if "discriminators" not in k:
                if k.startswith('module.'):
                    new_state_dict[k[7+4:]] = v  # remove 'module.' prefix
                else:
                    new_state_dict[k[4:]] = v
        self.dnn.load_state_dict(new_state_dict, strict=False)
        self.dnn.eval()
        print('have load score model!')
        

class SinModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--opt_type", type=str, choices=['Adam', 'AdamW'], required=False, default='AdamW',
                            help='The optimizer type.')
        parser.add_argument("--lr", type=float, required=False, default=8e-5, 
                            help="The learning rate for distillation")
        parser.add_argument("--beta1", type=float, default=0.8,
                            help="Beta1 for Adam/AdamW.")
        parser.add_argument("--beta2", type=float, default=0.99,
                            help="Beta2 for Adam/AdamW.")
        parser.add_argument("--ema_decay", type=float, default=0.999, 
                            help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03,
                            help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20,
                            help="Number of files for performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--teacher_ckp_path", type=str, required=False, default="XXX.ckpt",
                            help="The ckp path of the teacher model for distillation.")
        parser.add_argument("--teacher_inference_N", type=int, required=False, default=16,
                            help="The inference step for teacher model.")
        parser.add_argument("--loss_type_list", type=str, required=False, default="score_mse:1.0,multi-mel:0.1",
                            help="The type of spectral-related loss functions except distillation losses.")
        parser.add_argument("--distill_loss_type_list", type=str, required=False, default="distill,inverse,consistency",
                            help="The type of bijective loss functions for one-step distillation.")
        parser.add_argument("--use_omni_for_distill", action="store_false",
                            help="Whether to use omnidirectional format for distillation loss.")
        parser.add_argument("--use_gan", action='store_false',
                            help="Whether to use adversarial training.")
        parser.add_argument("--gan_weight", type=float, required=False, default=20.0,
                            help="Weight of the adversarial loss, only valid when use_gan=True.")
        return parser

    def __init__(self,
        backbone: str = "blade",
        sde: str = "ouvesde",
        opt_type: str = 'AdamW',
        beta1: float = 0.9,
        beta2: float = 0.99,
        lr: float = 1e-4,
        ema_decay: float = 0.999,
        t_eps: float = 3e-2,
        nolog: bool = False,
        num_eval_files: int = 20,
        teacher_ckp_path: str = "",
        loss_type_list: str = "",
        distill_loss_type_list: str = "",
        teacher_inference_N: int = 12,
        use_omni_for_distill: bool = False,
        use_gan: str = "true",
        gan_weight: float = 1.0,
        data_module_cls = None, 
        max_epochs = 3100,
        **kwargs
    ):
        """
        Create a new SinModel.
        """
        super().__init__()

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.sde_name = sde
        self.teacher_sde = sde_cls(**kwargs)
        # for determinstic inference
        self.teacher_sde.sampling_type = "ode_first_order"
        self.teacher_sde.N = teacher_inference_N

        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        self.teacher_dnn = dnn_cls(**kwargs)

        self.kwargs = kwargs
        self.max_epochs = max_epochs
        self.use_gan = use_gan
        self.gan_weight = gan_weight
        
        self.teacher_ckp_path = teacher_ckp_path
        self.use_omni_for_distill = use_omni_for_distill

        # load teacher weights
        try:
            self.load_teacher_model()
        except: 
            print("Loading Teacher model fails, please check teacher_ckp_path carefully!")
            pass
        
        # Store hyperparams and save them
        self.opt_type = opt_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps

        if isinstance(loss_type_list, str):
            if len(loss_type_list.strip()) < 1:  # ""
                loss_type_list = None
            else:
                loss_type_list = loss_type_list.strip().split(",")
        if isinstance(distill_loss_type_list, str):
            distill_loss_type_list = distill_loss_type_list.strip().split(",")
        
        self.loss_type_list = loss_type_list
        self.distill_loss_type_list = distill_loss_type_list

        last_epoch = -1
        if self.use_gan:
            print("Using GAN for one-step distillation!")
            self.mpd = MultiPeriodDiscriminator().to(self.device)
            self.mrd = MultiResolutionDiscriminator().to(self.device)

            try:
                mpd_weights, mrd_weights = OrderedDict(), OrderedDict()
                ckp = torch.load(self.teacher_ckp_path, map_location="cpu")["state_dict"]
                for k, v in ckp.items():
                    if k.startswith("mpd"):
                        mpd_weights[k[4:]] = v
                    elif k.startswith("mrd"):
                        mrd_weights[k[4:]] = v
                self.mpd.load_state_dict(mpd_weights)
                self.mrd.load_state_dict(mrd_weights)
                print("Discriminators have been loaded.")
            except:
                print("Did not search the weights for discriminators, so training from scratch.")
                pass

            if self.opt_type == "Adam":
                self.optim_d = torch.optim.Adam(itertools.chain(self.mpd.parameters(), self.mrd.parameters()), lr=self.lr, betas=[self.beta1, self.beta2])
            elif self.opt_type == "AdamW":
                self.optim_d = torch.optim.AdamW(itertools.chain(self.mpd.parameters(), self.mrd.parameters()), lr=self.lr, betas=[self.beta1, self.beta2])

            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999, last_epoch=last_epoch) 

        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        #
        self._reduce_op_3 = lambda *args, **kwargs: torch.mean(torch.sum(dim=[1, 2], *args, **kwargs))
        self._reduce_op_4 = lambda *args, **kwargs: torch.mean(torch.sum(dim=[1, 2, 3], *args, **kwargs))

        self.nolog = nolog
        self.configure_losses()

    def configure_losses(self):
        # loss
        self.weight_dict = {}
        if self.loss_type_list is not None:
            self.loss_dict = {}
            for cur_loss_zip in self.loss_type_list:
                cur_loss, cur_weight = cur_loss_zip.split(":")
                cur_weight = float(cur_weight)
                self.weight_dict[cur_loss.lower()] = cur_weight
                if cur_loss.lower() == "mel":
                    self.loss_dict[cur_loss.lower()] = MelLoss(sampling_rate=self.data_module.sampling_rate)
                elif cur_loss.lower() == "multi-mel":
                    self.loss_dict[cur_loss.lower()] = MultiresolutionMelLoss(sampling_rate=self.data_module.sampling_rate)
                elif cur_loss.lower() == "score_mae":
                    self.loss_dict[cur_loss.lower()] = lambda x: self._reduce_op_4(torch.abs(x))
                elif cur_loss.lower() == "score_mse":
                    self.loss_dict[cur_loss.lower()] = lambda x: self._reduce_op_4(torch.square(x))
                else:
                    raise NotImplementedError
        else:
            self.loss_dict = None

        if self.use_omni_for_distill:
            self.distill_loss = OmniDistillLoss(mag_dist_type="L2")
        else:
            self.distill_loss = lambda x1, x2: self._reduce_op_4(torch.square(x1 - x2))

    def configure_optimizers(self):
        if self.opt_type.lower() == "adam":
            optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif self.opt_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # set scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()
        if self.use_gan:
            self.scheduler_d.step()

    def load_teacher_model(self):
        """
        Load teacher model and freeze the parameters
        """
        ckp = torch.load(self.teacher_ckp_path, map_location=self.device)["state_dict"]
        nn_weights = OrderedDict()
        for k, v in ckp.items():
            if k.startswith("dnn"):
                nn_weights[k[4:]] = v  # dnn.[xxx]
        self.teacher_dnn.load_state_dict(nn_weights)
        for param in self.teacher_dnn.parameters():
            param.requires_grad = False
        self.teacher_dnn.eval()
        # student model should also inhert the weights
        self.dnn.load_state_dict(nn_weights)
        self.teacher_dnn.to(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")
        if self.use_gan:
            self.mpd.load_state_dict(checkpoint["mpd"])
            self.mrd.load_state_dict(checkpoint["mrd"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        checkpoint["generator"] = self.dnn.state_dict()
        if self.use_gan:
            checkpoint["mpd"] = self.mpd.state_dict()
            checkpoint["mrd"] = self.mrd.state_dict()
            checkpoint["optim_d"] = self.optim_d.state_dict()
            checkpoint["scheduler_d"] = self.scheduler_d.state_dict()
    
    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, score_wav=None, x_wav=None):
        """
        score_wav: (B, L)
        x_wav: (B, L)
        x_mag: (B, F, T)
        score_pha: (B, F, T)
        x_pha: (B, F, T)
        """
        loss_val_dict = {}
        loss = 0.
        if self.loss_dict is not None:
            for k in self.loss_dict:
                if k.lower() in ["mel", "multi-mel"]:
                    #print(x_wav, score_wav)
                    cur_loss = self.loss_dict[k](x_wav, score_wav, self._reduce_op_3)
                    loss = loss + self.weight_dict[k] * cur_loss
                else:
                    raise NotImplementedError
                loss_val_dict[k] = cur_loss.item()
            return loss, loss_val_dict
        else:
            return None, None

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def forward(self, x, t, y, **kwargs):
        """
        x: (B, 2, F, T)
        y: (B, 2, F, T)
        t: (B,)
        """
        score = self.dnn(x, cond=y, time_cond=t)

        return score

    def _mag_spec_fwd(self, spec):
        """
        spec: (B, F, T)
        return: (B, F, T)
        """
        if self.data_module.transform_type == "exponent":
            if self.data_module.spec_abs_exponent != 1:
                spec = (spec ** self.data_module.spec_abs_exponent) * self.data_module.spec_factor
        elif self.data_module.transform_type == "log":
            spec = torch.log(1 + spec)
        elif self.data_module.transform_type == "none":
            spec = spec
        return spec

    def _mag_spec_back(self, spec):
        """
        spec: (B, F, T)
        return: (B, F, T)
        """
        if self.data_module.transform_type == "exponent":
            if self.data_module.spec_abs_exponent != 1:
                spec = (spec / self.data_module.spec_factor) ** (1 / self.data_module.spec_abs_exponent)
        elif self.data_module.transform_type == "log":
            spec = torch.exp(spec) - 1
        elif self.data_module.transform_type == "none":
            spec = spec
        return spec

    def _step(self, batch, batch_idx):
        """
        x: (B, 1, F-1, T)/(B, 1, F, T), target, complex-valued
        y: (B, 1, F-1, T)/(B, 1, F, T), inpt, complex-valued
        """
        x, y, x_audio = batch
        x = torch.cat([x.real, x.imag], dim=1)  # (B, 2, F, T)
        y = torch.cat([y.real, y.imag], dim=1)  # (B, 2, F, T)
        x_audio = x_audio.squeeze(1)  # (B, L)
        real_len = x_audio.shape[-1]

        # Update discriminator per 2-steps
        if batch_idx % 2 == 0 and self.use_gan:
            # t=1-\eps
            t = torch.ones(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False) * (1 - self.sde.offset)
            # At end-point no noise is added for bridge
            xt = y
            score = self(xt, t, y) # generator
            score_mag, score_pha = self._mag_spec_back(torch.norm(score, dim=1)), torch.atan2(score[:, -1], score[:, 0])
            if self.data_module.drop_last_freq:
                last_score_freq_mag, last_score_freq_pha = score_mag[:, -1, None], score_pha[:, -1, None]
                score_mag_ = torch.cat([score_mag, last_score_freq_mag], dim=1)
                score_pha_ = torch.cat([score_pha, last_score_freq_pha], dim=1)
            else:
                score_mag_, score_pha_ = score_mag, score_pha

            score_decom = torch.complex(score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_))
            score_wav = torch.istft(score_decom,
                                    n_fft=self.data_module.n_fft,
                                    hop_length=self.data_module.hop_size,
                                    win_length=self.data_module.win_size,
                                    window=torch.hann_window(self.data_module.win_size).to(score_mag_.device),
                                    center=True,
                                    length=real_len,
                                    ) # score_wav就是x_esti_audio
            self.optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(x_audio, score_wav.detach())
            loss_disc_f, _, _ = DiscriminatorLoss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = self.mrd(x_audio, score_wav.detach())
            loss_disc_s, _, _ = DiscriminatorLoss(y_ds_hat_r, y_ds_hat_g)
            L_D = loss_disc_s + loss_disc_f

            L_D.backward()
            self.optim_d.step()

            if self.use_gan:  # adjust lr for discriminators, here we use step-based scheduler_d for distillation
                self.scheduler_d.step()

        # Update generator
        # for teacher, conduct ODE sampling for deterministic estimation
        teacher_est = self.teacher_sde.reverse_diffusion(x1=y, cond=y, dnn=self.teacher_dnn, to_cpu=False).detach()
        tT = torch.ones(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False) * (1 - self.sde.offset)
        xt = y
        score_T_GT = self(xt, tT, y)  # f(xT, T, cond)->\hat{x0}, reconstruction
        score_0_teacher = self(teacher_est, 0 * tT, y)  # f(teacher_est, 0, cond)->\hat{xT}  # inverse
        score_0_GT = self(x, 0 * tT, y).detach()
        score_T_cons = self(score_0_GT, tT, y)  # f(f(x0, 0, cond), T, cond)->\hat{x0}  # consistency

        loss_G = 0.
        loss_dict_tot = {}

        # distillation-loss related
        if self.use_omni_for_distill:
            for k in self.distill_loss_type_list:
                if k.lower() == "distill":
                    tmp_loss = self.distill_loss(teacher_est, score_T_GT, self._reduce_op_4)
                    loss_G = loss_G + tmp_loss
                elif k.lower() == "inverse":
                    tmp_loss = self.distill_loss(y, score_0_teacher, self._reduce_op_4)
                    loss_G = loss_G + tmp_loss
                elif k.lower() == "consitency":
                    tmp_loss = self.distill_loss(x, score_T_cons, self._reduce_op_4)
                    loss_G = loss_G + tmp_loss
                loss_dict_tot[k] = tmp_loss.item()
        else:
            for k in self.distill_loss_type_list:
                if k.lower() == "distill":
                    tmp_loss = self.distill_loss(teacher_est, score_T_GT)
                    loss_G = loss_G + tmp_loss
                elif k.lower() == "inverse":
                    tmp_loss = self.distill_loss(y, score_0_teacher)
                    loss_G = loss_G + tmp_loss
                elif k.lower() == "consistency":
                    tmp_loss = self.distill_loss(x, score_T_cons)
                    loss_G = loss_G + tmp_loss
                loss_dict_tot[k] = tmp_loss.item()

        # other loss needed (optional)
        score_mag_orig = torch.norm(score_T_GT, dim=1)
        score_pha = torch.atan2(score_T_GT[:, -1], score_T_GT[:, 0])
        score_mag = self._mag_spec_back(score_mag_orig)
        if self.data_module.drop_last_freq:
            last_score_freq_mag, last_score_freq_pha = score_mag[:, -1, None], score_pha[:, -1, None]
            score_mag_ = torch.cat([score_mag, last_score_freq_mag], dim=1)
            score_pha_ = torch.cat([score_pha, last_score_freq_pha], dim=1)
        else:
            score_mag_, score_pha_ = score_mag, score_pha

        score_decom = torch.complex(score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_))
        score_wav = torch.istft(score_decom,
                                n_fft=self.data_module.n_fft,
                                hop_length=self.data_module.hop_size,
                                win_length=self.data_module.win_size,
                                window=torch.hann_window(self.data_module.win_size).to(score_mag_.device),
                                center=True,
                                length=real_len,
                                ) # score_wav就是x_esti_audio
        loss_op, loss_val_dict = self._loss(score_wav=score_wav, x_wav=x_audio)
        if loss_op is not None and loss_val_dict is not None:
            loss_G = loss_G + loss_op
            loss_dict_tot.update(loss_val_dict)        
        
        # whether to use gan-related generator loss (optional)
        if self.use_gan:
            _, y_df_g, fmap_f_r, fmap_f_g = self.mpd(x_audio, score_wav)
            _, y_ds_g, fmap_s_r, fmap_s_g = self.mrd(x_audio, score_wav)
            loss_fm_f = FeatureMatchingLoss(fmap_f_r, fmap_f_g)
            loss_fm_s = FeatureMatchingLoss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = GeneratorLoss(y_df_g)
            loss_gen_s, _ = GeneratorLoss(y_ds_g)
            L_GAN_G = loss_gen_s + loss_gen_f
            L_FM = loss_fm_s + loss_fm_f
            loss_G = loss_G + self.gan_weight * (L_GAN_G + L_FM)

        return loss_G, loss_dict_tot

    def training_step(self, batch, batch_idx, **kwargs):
        loss, loss_val_dict = self._step(batch, batch_idx)
        for k in loss_val_dict:
            self.log(f'train_loss_{k}', loss_val_dict[k], on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss, loss_val_dict = self._step(batch, 1)  # batch_idx=1不更新L_D.backward()
        for k in loss_val_dict:
            self.log(f'valid_loss_{k}', loss_val_dict[k], on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, estoi_est, periodicity_est = evaluate_sin_model(self, self.num_eval_files)
            print(f"PESQ at epoch {self.current_epoch}: {pesq_est:.3f}")
            print(f"ESTOI at epoch {self.current_epoch}: {estoi_est:.3f}")
            print(f"Periodicity at epoch {self.current_epoch} : {periodicity_est:.4f}")
            print('__________________________________________________________________')
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)
            self.log('ValidationPeriodicity', periodicity_est, on_step=False, on_epoch=True)

        return loss
    
    def validation_epoch_end(self, outputs):
        if self.use_gan:
            discriminator_ckp_path = os.path.join(self.logger.log_dir, f"discriminator_epoch_{self.current_epoch}.pt")
            torch.save({
                "mpd": self.mpd.state_dict(),
                "mrd": self.mrd.state_dict()
            }, discriminator_ckp_path)

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, sig):
        x = self._inv_mel(self._mel(sig))
        return x

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)
    
    def _mel(self, sig):
        return self.data_module.sig2mel(sig)
    
    def _inv_mel(self, mel):
        return self.data_module.inv_mel(mel)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def load_score_model(self, checkpoint):
        #self.score_net = ScoreModel.load_from_checkpoint(checkpoint).dnn
        from collections import OrderedDict
        import io
        if isinstance(checkpoint, dict):
            # 如果 checkpoint 是一个字典，先将其序列化为字节流
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            state_dict = torch.load(buffer, map_location=self.device)
        else:
            # 否则，直接加载
            state_dict = torch.load(checkpoint, map_location=self.device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            if "discriminators" not in k:
                if k.startswith('module.'):
                    new_state_dict[k[7+4:]] = v  # remove 'module.' prefix
                else:
                    new_state_dict[k[4:]] = v
        self.dnn.load_state_dict(new_state_dict, strict=False)
        self.dnn.eval()
        print('have load score model!')
