"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import torch.nn as nn
import warnings
import math
import numpy as np
from div.util.tensors import batch_broadcast
import torch
from div.util.registry import Registry
from scipy import integrate
from librosa.filters import mel as librosa_mel_fn

SDERegistry = Registry("SDE")

class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    # @abc.abstractmethod
    # def prior_sampling(self, shape, *args):
    #     """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
    #     pass

    # @abc.abstractmethod
    # def prior_logp(self, z):
    #     """Compute log-density of the prior distribution.

    #     Useful for computing the log-likelihood via probability flow ODE.

    #     Args:
    #         z: latent code
    #     Returns:
    #         log probability density
    #     """
    #     pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False, diffusion_power_gradient=None):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
            diffusion_power_gradient: func or None. if the diffusion function has a gradient with respect to the process X, we need to include it in the reverse SDE
            (cf Anderson1982 or Appendix A of Song2021)
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
                self.diffusion_power_gradient = diffusion_power_gradient

            @property
            def T(self):
                return T

            def sde(self, x, t, *args, **kwargs):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args, **kwargs)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args, **kwargs):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                if "conditioning" in kwargs.keys() and kwargs["conditioning"] is not None:
                    score = score_model(x, t, kwargs["conditioning"])
                else:
                    score = score_model(x, t, *args)
                # raw_dnn_output = score_model._raw_dnn_output(x, t, *args)
                if sde_diffusion.ndim < x.ndim:
                    sde_diffusion = sde_diffusion.view(*sde_diffusion.size(), *((1,)*(x.ndim - sde_diffusion.ndim)))
                score_drift = -sde_diffusion**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                if diffusion_power_gradient is not None:
                    total_drift -= diffusion_power_gradient(x, t)
                # return {
                #     'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                #     'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                #     'raw_dnn_output': raw_dnn_output
                # }
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score
                }

            def discretize(self, x, t, *args, **kwargs):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args)
                if G.ndim < x.ndim:
                    G = G.view(*G.size(), *((1,)*(x.ndim - G.ndim)))
                if "conditioning" in kwargs.keys() and kwargs["conditioning"] is not None:
                    rev_f = f - G ** 2 * score_model(x, t, score_conditioning=kwargs["conditioning"], sde_input=args[0]) * (0.5 if self.probability_flow else 1.)
                else:
                    rev_f = f - G ** 2 * score_model(x, t, *args) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass

@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--theta", type=float, default=1.5, 
            help="The constant stiffness of the Ornstein-Uhlenbeck process.")
        parser.add_argument("--sigma-min", type=float, required=True, default=0.05, 
            help="The minimum sigma to use.")
        parser.add_argument("--sigma-max", type=float, required=True, default=0.5, 
            help="The maximum sigma to use.")
        parser.add_argument("--snr", type=float, required=True, default=0.5)
        parser.add_argument("--reverse_n", type=int, required=True, default=10,
            help="Timesteps for reverse.")
        parser.add_argument("--corrector_steps", type=int, required=True, default=1,
            help="Steps for corrector.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=1000, snr=0.5, reverse_n=10, corrector_steps=1, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self.snr = snr
        self.reverse_n = reverse_n
        self.corrector_steps = corrector_steps

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t, **kwargs):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        t = t[:, None, None, None]
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        return y + torch.randn_like(y) * std[:, None, None, None]
    
    def forward_diffusion(self, x0, x1, t):
        mean_, std_ = self.marginal_prob(x0, t, x1)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean_ + std_ * z
        xt.detach()
        target = z

        return xt, target, std_

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")



@SDERegistry.register("bridgegan")
class BridgeGAN(SDE, nn.Module):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--beta_min", type=float, required=True, default=0.01,
                            help="Beta min")
        parser.add_argument("--beta_max", type=float, required=True, default=2,
                            help="Beta max")
        parser.add_argument("--c", type=float, required=True, default=0.4,
                            help="Noise scheduler parameter.")
        parser.add_argument("--k", type=float, required=True, default=2.6,
                            help="Noise scheduler parameter.")
        parser.add_argument("--bridge_type", type=str, required=True, default="gmax",
                            choices=["vp", "ve", "gmax"],
                            help="Type of bridge diffusion.")
        parser.add_argument("--N", type=int, required=True, default=4,
                            help="Number of sampling in the reverse.")
        parser.add_argument("--offset", type=float, default=1e-5,
                            help="Offset for time discrete.")
        parser.add_argument("--predictor", type=str, required=True, default="x0", choices=["x0", "hpsi"],
                            help="Type of training object.")
        parser.add_argument("--sampling_type", type=str, required=True, default="sde_first_order", 
                            choices=["sde_first_order", "ode_first_order"],
                            help="Sampling type in the inference.")
        parser.add_argument("--noise_tune", action='store_true', # store_true：如果指定了这个参数，值为 True，否则为 False。
                            help="if add noise using (1-A^T*A) to tune.")
        #parser.add_argument("--sampling_rate", type=float, required=False, default=22050)
        return parser

    def __init__(self, 
                 beta_min=0.01, 
                 beta_max=20, 
                 c=0.4, 
                 k=2.6, 
                 bridge_type="vp", 
                 N=10,
                 offset=1e-5,
                 predictor="x0",
                 sampling_type="sde_first_order",
                 noise_tune = False,
                 **ignored_kwargs
                 ):
        #super().__init__(N)
        SDE.__init__(self, N)
        nn.Module.__init__(self)
        self.beta_min = beta_min
        self.beta_max = beta_max
        print('beta_max=',beta_max)
        self.c = c
        self.k = k
        self.bridge_type = bridge_type
        self.N = N
        self.offset = offset
        self.predictor = predictor
        self.sampling_type = sampling_type
        self.noise_tune = noise_tune
        # print(self.noise_tune)
        # exit()
        if self.noise_tune:
            print("noise_tune yes!")
            # mel matrix
            mel = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
            mel_basis = torch.from_numpy(mel)
            inv_mel_basis = mel_basis.pinverse() # 伪逆
            PhiTPhi = inv_mel_basis@mel_basis
            # Phi:(Fmel,F)，PhiT:(F,Fmel)
            # self.register_buffer('Phi', mel_basis)
            # self.register_buffer('PhiT', inv_mel_basis)
            # self.register_buffer('PhiTPhi', inv_mel_basis@mel_basis)
            null_filter = (torch.eye(1024//2+1, dtype=PhiTPhi.dtype, device=PhiTPhi.device) - PhiTPhi)[:-1, :-1]# [256,256]
            self.register_buffer('null_filter', null_filter)

    def get_alpha(self, t):
        """
        return: (B, 1, 1, 1)
        """
        if self.bridge_type.lower() == "vp":
            return torch.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)))
        elif self.bridge_type.lower() in ["ve", "gmax"]:
            return 1
    
    def get_bar_alpha(self, t):
        """
        return: (B, 1, 1, 1)
        """
        if self.bridge_type.lower() == "vp":
            out = torch.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2) - (self.beta_min + 0.5 * (self.beta_max - self.beta_min))))
        elif self.bridge_type.lower() in ["ve", "gmax"]:
            out = 1
        return out
    
    def get_sigma2(self, t):
        """
        return: (B, 1, 1, 1)
        """
        if self.bridge_type.lower() == "vp":
            sigma2 = self.c * (torch.exp(self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)) - 1)
        elif self.bridge_type.lower() == "ve":
            sigma2 = self.c * (self.k ** (2 * t) - 1) / (2 * math.log(self.k))
        elif self.bridge_type.lower() == "gmax":
            sigma2 = self.beta_min * t + (self.beta_max - self.beta_min) * (t ** 2) / 2
        return sigma2
    
    def get_bar_sigma2(self, t):
        """
        return: (B, 1, 1, 1)
        """
        if self.bridge_type.lower() == "vp":
            sigma2 = self.c * (math.exp(self.beta_min + 0.5 * (self.beta_max - self.beta_min)) - torch.exp(self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)))
        elif self.bridge_type.lower() == "ve":
            sigma2 = self.c * (self.k ** 2 - self.k ** (2 * t)) / (2 * math.log(self.k))
        elif self.bridge_type.lower() == "gmax":
            sigma2 = (self.beta_min + self.beta_max) / 2 - (self.beta_min * t + (self.beta_max - self.beta_min) * (t ** 2) / 2)
        return sigma2

    @property
    def T(self):
        return self.N
    
    def copy(self):
        return BridgeGAN(beta_min=self.beta_min, 
                         beta_max=self.beta_max,
                         c=self.c,
                         k=self.k,
                         bridge_type=self.bridge_type,
                         N=self.N,
                         offset=self.offset,
                         predictor=self.predictor,
                         sampling_type=self.sampling_type)
    
    def sde(self, x, t, *args):
        pass

    def marginal_prob(self, x0, x1, t):
        """
        x0: (B, 2, F, T)
        x1: (B, 2, F, T)
        t: (B,)
        """
        t = t[:, None, None, None]  # (B, 1, 1, 1)
        alpha_t = self.get_alpha(t)
        bar_alpha_t = self.get_bar_alpha(t)
        sigma2_t = self.get_sigma2(t)
        bar_sigma2_t = self.get_bar_sigma2(t)
        mean_ = (alpha_t * bar_sigma2_t * x0 + bar_alpha_t * sigma2_t * x1) / (bar_sigma2_t + sigma2_t)
        std_ = torch.sqrt((alpha_t ** 2 * sigma2_t * bar_sigma2_t) / (bar_sigma2_t + sigma2_t))

        return mean_, std_, alpha_t, bar_alpha_t, sigma2_t, bar_sigma2_t

    def forward_diffusion(self, x0, x1, t):
        mean_, std_, alpha_t, bar_alpha_t, sigma2_t, bar_sigma2_t = self.marginal_prob(x0, x1, t)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        if self.noise_tune:
            #print("noise_tune yes1!")
            z = torch.einsum('fk,bhkt->bhft', [self.null_filter.to(x0.device), z])
        xt = mean_ + std_ * z
        xt.detach()

        if self.predictor.lower() == "hpsi":
            target = (xt - alpha_t * x0) / (alpha_t * torch.sqrt(sigma2_t))
            # Control the variance of the input to 1
            xt = (xt - alpha_t / self.alpha_1 * x1) / (alpha_t * torch.sqrt(bar_sigma2_t))
            xt.detach()
            target.detach()
        
        elif self.predictor.lower() == "x0":
            target = x0
            target.detach()

        else:
            raise NotImplementedError(F"Unsupported predictor {self.predictor}.")
        target.detach()
        return xt, target

    @torch.no_grad()
    def data_estimation(self, xt, x1, cond, t, dnn):
        """
        xt: (B, 2, F, T)
        x1: (B, 2, F, T)
        cond: score_conditioning
        t: (B,)
        dnn: model
        """
        t = t[:, None, None, None]

        if self.predictor.lower() == "hpsi":
            alpha_t = self.get_alpha(t)
            sigma_t = torch.sqrt(self.get_sigma2(t))
            bar_sigma_t = torch.sqrt(self.get_bar_sigma2(t))

            xt_input = (xt - alpha_t / self.alpha_1 * x1) / (alpha_t * bar_sigma_t)
            eps_t = dnn(xt_input, x1, t[:, 0, 0, 0])
            hat_x0 = xt / alpha_t - sigma_t * eps_t

        elif self.predictor.lower() == "x0":
            hat_x0 = dnn(xt, x1, t[:, 0, 0, 0])
        
        else:
            raise NotImplementedError(f"Unsupported predictor {self.predictor}")
        
        return hat_x0

    @torch.no_grad()
    def reverse_diffusion(self, x1, cond, dnn):
        """
        x1: (B, 2, F, T)
        cond: score_conditioning
        return: (B, 2, F, T)
        """
        h = 1.0 / self.N
        xt = x1
        xt_traj = [xt.detach().cpu()]
        for i in range(self.N):
            if self.sampling_type.lower() == "sde_first_order":
                # s -> t
                s = (1.0 - self.offset - (i * h)) * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)
                t = max(self.offset, 1.0 - self.offset - (i+1)*h) * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)
                # Prepare all needed variables
                sigma2_t = self.get_sigma2(t)
                sigma2_s = self.get_sigma2(s)
                alpha_t = self.get_alpha(t)
                alpha_s = self.get_alpha(s)

                xs = xt
                hat_x0 = self.data_estimation(xs, x1, cond, s, dnn)

                coeff = (sigma2_t) / (sigma2_s)
                xt = (alpha_t / alpha_s) * coeff * xs + alpha_t * (1 - coeff) * hat_x0
                if i != self.N - 1:
                    eps = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, requires_grad=False)
                    if self.noise_tune:
                        #print("noise_tune yes2!")
                        eps = torch.einsum('fk,bhkt->bhft', [self.null_filter.to(x1.device), eps])
                    xt += alpha_t * torch.sqrt(sigma2_t * (1 - coeff)) * eps

            elif self.sampling_type.lower() == "ode_first_order":
                # s -> t
                s = (1.0 - self.offset - (i * h)) * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)
                t = max(self.offset, 1.0 - self.offset - (i + 1) * h) * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)
                # Prepare all needed variables
                sigma2_t = self.get_sigma2(t)
                bar_sigma2_t = self.get_bar_sigma2(t)
                sigma2_s = self.get_sigma2(s)
                bar_sigma2_s = self.get_bar_sigma2(s)
                sigma_t = torch.sqrt(sigma2_t)
                bar_sigma_t = torch.sqrt(bar_sigma2_t)
                sigma_s = torch.sqrt(sigma2_s)
                bar_sigma_s = torch.sqrt(bar_sigma2_s)
                sigma2_1 = sigma2_t + bar_sigma2_t 
                alpha_t = self.get_alpha(t)
                alpha_s = self.get_alpha(s)

                xs = xt
                hat_x0 = self.data_estimation(xs, x1, cond, s, dnn)

                xt = (alpha_t * sigma_t * bar_sigma_t) / (alpha_s * sigma_s * bar_sigma_s) * xs + alpha_t / sigma2_1 * ((bar_sigma2_t - (bar_sigma_s * sigma_t * bar_sigma_t) / (sigma_s)) * hat_x0 + (sigma2_t - (sigma_s * sigma_t * bar_sigma_t) / (bar_sigma_s)) * x1 / self.alpha_1)
            
            else:
                raise NotImplemented()
            
            xt_traj.append(xt.detach().cpu())
        
        xt_traj = torch.stack(xt_traj, dim=1)

        return xt_traj[:, -1] 
