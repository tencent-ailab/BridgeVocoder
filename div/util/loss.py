import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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


class MultiresolutionMelLoss(nn.Module):
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
        super(MultiresolutionMelLoss, self).__init__()
        self.resolutions = resolutions
        self.sampling_rate = sampling_rate

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, op=None):
        """
        y: (B, L)
        y_hat: (B, L)
        """
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
            loss_tot = loss_tot + op(torch.abs(y_mel - y_hat_mel))
        loss_tot = loss_tot / len(self.resolutions)
        return loss_tot


class MelLoss(nn.Module):
    def __init__(self, 
                 resolution=(1024, 256, 1024, 80),
                 sampling_rate=22050,
        ):
        super(MelLoss, self).__init__()
        self.resolution = resolution
        self.sampling_rate = sampling_rate
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, op=None):
        """
        y: (B, L)
        y_hat: (B, L)
        """
        y_mel = mel_spectrogram(y, 
                                n_fft=self.resolution[0],
                                num_mels=self.resolution[-1],
                                sampling_rate=self.sampling_rate,
                                hop_size=self.resolution[1],
                                win_size=self.resolution[2],
                                fmin=0,
                                fmax=self.sampling_rate / 2,
                                )
        y_hat_mel = mel_spectrogram(y_hat, 
                                n_fft=self.resolution[0],
                                num_mels=self.resolution[-1],
                                sampling_rate=self.sampling_rate,
                                hop_size=self.resolution[1],
                                win_size=self.resolution[2],
                                fmin=0,
                                fmax=self.sampling_rate / 2,
                                )
        loss = op(torch.abs(y_mel - y_hat_mel))
        return loss


class OmniDistillLoss(nn.Module):
    def __init__(self, mag_dist_type="l2", order=1):
        super(OmniDistillLoss, self).__init__()
        self.mag_dist_type = mag_dist_type
        self.order = order

        kernel1 = torch.from_numpy(np.array([[-1., 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32'))
        kernel2 = torch.from_numpy(np.array([[0, -1., 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel3 = torch.from_numpy(np.array([[0, 0, -1.], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel4 = torch.from_numpy(np.array([[0, 0, 0], [-1., 1., 0], [0, 0, 0]], dtype='float32'))
        kernel5 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel6 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., -1.], [0, 0, 0]], dtype='float32'))
        kernel7 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [-1., 0, 0]], dtype='float32'))
        kernel8 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, -1., 0]], dtype='float32'))
        kernel9 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, -1.]], dtype='float32'))
        kernels = torch.stack([kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9], dim=0)  # (out_nch, 3, 3)
        kernels = kernels.unsqueeze(1)
        self.filters = kernels
    
    def forward(self, y, y_g, op=None):
        """
        y: (B, 2, F, T)
        y_g: (B, 2, F, T)
        """
        mag, pha = torch.norm(y, dim=1).unsqueeze(1), torch.atan2(y[:, -1], y[:, 0]).unsqueeze(1)
        mag_g, pha_g = torch.norm(y_g, dim=1).unsqueeze(1), torch.atan2(y_g[:, -1], y_g[:, 0]).unsqueeze(1)
        
        pha = F.conv2d(pha, self.filters.to(pha.device), bias=None, stride=1, padding=1)
        pha_g = F.conv2d(pha_g, self.filters.to(pha_g.device), bias=None, stride=1, padding=1)
        cur_rea, cur_rea_g = mag.repeat(1, self.filters.shape[0], 1, 1) * torch.cos(pha), \
                             mag_g.repeat(1, self.filters.shape[0], 1, 1) * torch.cos(pha_g)
        cur_imag, cur_imag_g = mag.repeat(1, self.filters.shape[0], 1, 1) * torch.sin(pha), \
                               mag_g.repeat(1, self.filters.shape[0], 1, 1) * torch.sin(pha_g)

        if self.mag_dist_type.upper() == "L1":
            loss_R, loss_I = torch.abs(cur_rea - cur_rea_g), torch.abs(cur_imag - cur_imag_g)
        elif self.mag_dist_type.upper() == "L2":
            loss_R, loss_I = torch.square(cur_rea - cur_rea_g), torch.square(cur_imag - cur_imag_g)
        loss_R, loss_I = torch.nan_to_num(loss_R), torch.nan_to_num(loss_I)
        loss = op(0.5 * (loss_R + loss_I) / 9)
        return loss


def DiscriminatorLoss(disc_real_outputs, disc_generated_outputs): # 更新mpd和mrd
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(torch.clamp(1 - dr, min=0))
        g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        loss = loss + (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def GeneratorLoss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(torch.clamp(1 - dg, min=0))
        gen_losses.append(l)
        loss = loss + l

    return loss, gen_losses

def FeatureMatchingLoss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))

    return loss