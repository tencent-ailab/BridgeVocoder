import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Resample
from librosa.filters import mel as librosa_mel_fn
from typing import *
import numpy as np

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_window = {}
inv_mel_window = {}


def param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"

def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=True,
    in_dataset=False,
):
    global mel_window
    device = torch.device("cpu") if in_dataset else y.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in mel_window:
        mel_basis, hann_window = mel_window[ps]
        # print(mel_basis, hann_window)
        # mel_basis, hann_window = mel_basis.to(y.device), hann_window.to(y.device)
    else:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        mel_window[ps] = (mel_basis.clone(), hann_window.clone())

    spec = torch.stft(
        y.to(device),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window.to(device),
        center=True,
        return_complex=True,
    )

    spec = mel_basis.to(device) @ spec.abs()
    spec = spectral_normalize_torch(spec)

    return spec  # [batch_size,n_fft/2+1,frames]

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        4,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        4,
                        16,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        16,
                        64,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        64,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(128, 128, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(128, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        if x.ndim == 2:
            x= x.unsqueeze(1)  # (B, 1, L)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, mpd_reshapes=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(mpd_reshapes[0]),
                DiscriminatorP(mpd_reshapes[1]),
                DiscriminatorP(mpd_reshapes[2]),
                DiscriminatorP(mpd_reshapes[3]),
                DiscriminatorP(mpd_reshapes[-1]),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Multi-scale mel loss
class MultiResolutionMelLoss(nn.Module):
    def __init__(self,
                 resolutions=((32, 8, 32, 5),
                              (64, 16, 64, 10),
                              (128, 32, 128, 20),
                              (256, 64, 256, 40),
                              (512, 128, 512, 80),
                              (1024, 256, 1024, 160),
                              (2048, 512, 2048, 320),
                              ),
                sampling_rate=24000,
    ):
        super(MultiResolutionMelLoss, self).__init__()
        self.resolutions = resolutions
        self.sampling_rate = sampling_rate
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        loss_tot = 0.
        for i, cur_reso in enumerate(self.resolutions):
            y_mel = mel_spectrogram(y, 
                                    n_fft=cur_reso[0], 
                                    num_mels=cur_reso[-1],
                                    sampling_rate=self.sampling_rate,
                                    hop_size=cur_reso[1],
                                    win_size=cur_reso[2],
                                    fmin=0,
                                    fmax=self.sampling_rate / 2,
                                    )
            y_hat_mel = mel_spectrogram(y_hat, 
                                        n_fft=cur_reso[0], 
                                        num_mels=cur_reso[-1],
                                        sampling_rate=self.sampling_rate,
                                        hop_size=cur_reso[1],
                                        win_size=cur_reso[2],
                                        fmin=0,
                                        fmax=self.sampling_rate / 2,
                                        )
            loss_tot = loss_tot + F.l1_loss(y_mel, y_hat_mel)
        loss_tot = loss_tot / len(self.resolutions)
        return loss_tot



class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        num_embeddings: int = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(resolution=r, num_embeddings=num_embeddings)
                for r in resolutions
            ]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution,
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=(7, 5),
                        stride=(2, 2),
                        padding=(3, 2),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 2),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 1), padding=1
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 2), padding=1
                    )
                ),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=channels
            )
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        fmap = []
        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).abs()

        return magnitude_spectrogram

