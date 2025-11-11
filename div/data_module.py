import os
import math
import torch
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from librosa.filters import mel as librosa_mel_fn
import librosa
from typing import *
import numpy as np
import torch.nn.functional as F


def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")

def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

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

def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    in_dataset=False,
):
    global inv_mel_window, mel_window
    device = torch.device("cpu") if in_dataset else mel.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in inv_mel_window:
        inv_basis = inv_mel_window[ps]
    else:
        if ps in mel_window:
            mel_basis, _ = mel_window[ps]
        else:
            mel_np = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis = torch.from_numpy(mel_np).float().to(device)
            hann_window = torch.hann_window(win_size).to(device)
            mel_window[ps] = (mel_basis.clone(), hann_window.clone())
        inv_basis = mel_basis.pinverse()
        inv_mel_window[ps] = inv_basis.clone()
    return inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))


def amp_pha_specturm(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size).to(y.device)

    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )  # [batch_size, n_fft//2+1, frames, 2]

    log_amplitude = torch.log(
        stft_spec.abs() + 1e-5
    )  # [batch_size, n_fft//2+1, frames]
    phase = torch.atan2(stft_spec.imag, stft_spec.real)  # [batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, stft_spec.real, stft_spec.imag


class Specs(Dataset):
    def __init__(self, 
                 raw_wavfile_path, 
                 data_dir, 
                 subset, 
                 shuffle_spec, 
                 num_frames, 
                 sampling_rate,
                 n_fft, 
                 num_mels, 
                 hop_size, 
                 win_size, 
                 fmin, 
                 fmax,
                 format='default', 
                 phase_init='zero', 
                 normalize=True, 
                 spec_transform=None, 
                 drop_last_freq=True,
                 stft_kwargs=None,
                 ):

        # Read file paths according to file naming format.
        if format == "default":
            self.wav_files = []
            lines = open(data_dir, 'r').readlines()
            for l in lines:
                cur_filename = l.strip().split('/')[1].split('|')[0]
                self.wav_files.append(os.path.join(raw_wavfile_path, cur_filename))
            if subset == 'train':
                # shuffle the training files
                random.seed(1234)
                random.shuffle(self.wav_files)
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.phase_init = phase_init
        self.drop_last_freq = drop_last_freq
        self.subset = subset

    def __getitem__(self, i):
        if self.subset == 'train':
            i = np.random.randint(0, len(self.wav_files) - 1)# i = np.random.randint(0, len(self.wav_files) - 1)
        filename = self.wav_files[i]
        audio = load_wav(filename, self.sampling_rate)
        audio = torch.FloatTensor(audio).unsqueeze(0)  # (1, T)

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_size
        current_len = audio.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len-target_len)/2)
            audio = audio[..., start : start + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            audio = F.pad(audio, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
        
        if audio.abs().sum() < 1e-8 :
            print('Skip the previous get_item func.')
            # if len(self) > 1:
            #     i = np.random.randint(0, len(self.wav_files) - 1)
            return self[i]
        # normalize w.r.t to the signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize:
            normfac = torch.max(torch.abs(audio)) + 1e-6
        else:
            normfac = 1.0

        audio = audio / normfac

        mel = mel_spectrogram(audio,
                              n_fft=self.n_fft,
                              num_mels=self.num_mels,
                              sampling_rate=self.sampling_rate,
                              hop_size=self.hop_size,
                              win_size=self.win_size,
                              fmin=self.fmin,
                              fmax=self.fmax,
                              center=True,
                              in_dataset=True)
        # apply inv-mel
        inv_mel = inverse_mel(mel,
                              n_fft=self.n_fft,
                              num_mels=self.num_mels,
                              sampling_rate=self.sampling_rate,
                              hop_size=self.hop_size,
                              win_size=self.win_size,
                              fmin=self.fmin,
                              fmax=self.fmax,
                              in_dataset=True)

        inv_mel = inv_mel.abs().clamp_min_(1e-6)
        log_inv_mel = inv_mel.log()
        if self.phase_init == 'random':
            phase_ = 2 * math.pi * torch.rand_like(inv_mel) - math.pi  # [-pi, pi) 
        elif self.phase_init == 'zero':
            phase_ = torch.zeros_like(inv_mel)
        Y = torch.complex(inv_mel * torch.cos(phase_), inv_mel * torch.sin(phase_))  # (B, F, T)

        X = torch.stft(audio,
                       n_fft=self.n_fft,
                       hop_length=self.hop_size,
                       win_length=self.win_size,
                       window=torch.hann_window(self.win_size).to(audio.device),
                       center=True,
                       return_complex=True,
                       )
        X_compress, Y_compress = self.spec_transform(X), self.spec_transform(Y)  # complex-tensor, (B, F, T)
        if self.drop_last_freq:
            X_compress, Y_compress = X_compress[:, :-1].contiguous(), Y_compress[:, :-1].contiguous()  # (B, F-1, T)

        return X_compress, Y_compress, audio

    def __len__(self):
        return len(self.wav_files) if self.subset=='train' else len(self.wav_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--dataset_name", type=str, required=True, default="",
                            help="Name of the used dataset, for example, LJSpeech, LibriTTS...")
        parser.add_argument("--raw_wavfile_path", type=str, required=True, default="", 
                            help="The base directory of the raw wavfiles.")
        parser.add_argument("--train_data_dir", type=str, required=True, default="",
                             help="The scp path of the training dataset")
        parser.add_argument("--val_data_dir", type=str, required=True, default="", 
                            help="The scp path of the validation dataset")
        parser.add_argument("--test_data_dir", type=str, default="./LSJ/ljs_audio_text_test_filelist.txt", 
                            help="The scp path of the testing dataset")
        parser.add_argument("--format", type=str, choices=["default", "reverb"], default="default", 
                            help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, required=True, default=2, 
                            help="The batch size. 8 by default.")
        parser.add_argument("--sampling_rate", type=int, required=True, default=22050, 
                            help="Sampling rate.")
        parser.add_argument("--n_fft", type=int, required=True, default=1024, 
                            help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--num_mels", type=int, required=True, default=80, 
                            help="Number of mels.")
        parser.add_argument("--hop_size", type=int, required=True, default=256, 
                            help="Window hop length. 128 by default.")
        parser.add_argument("--win_size", type=int, required=True, default=1024, 
                            help="Window size, 1024 by default.")
        parser.add_argument("--fmin", type=int, default=0, 
                            help="Minimum frequency for mel conversion.")
        parser.add_argument("--fmax", type=int, required=True, default=8000, 
                            help="Maximum frequency for mel conversion.")
        parser.add_argument("--num_frames", type=int, required=True, default=256, 
                            help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--phase_init", type=str, choices=["random", "zero"], default="zero", 
                            help="Phase initization method.")
        parser.add_argument("--num_workers", type=int, required=True, default=4, 
                            help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--spec_factor", type=float, required=True, default=0.33, 
                            help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, required=True, default=0.5, 
                            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=bool, default=True, 
                            help="Whether to apoply the normalization strategy.")
        parser.add_argument("--transform_type", type=str, choices=["exponent", "log", "none"], default="exponent", 
                            help="Spectogram transformation for input representation.")
        parser.add_argument("--drop_last_freq", type=bool, default=True,
                            help="Whether to drop the last frequency band to meet the exp(2) requirement.")
        return parser

    def __init__(
        self, 
        raw_wavfile_path, 
        train_data_dir, 
        val_data_dir, 
        test_data_dir, 
        format='default', 
        batch_size=8,
        sampling_rate=22050, 
        n_fft=1024, 
        num_mels=80, 
        hop_size=256, 
        win_size=1024, 
        fmin=0,
        fmax=8000,
        num_frames=256,
        phase_init="zero",
        num_workers=4,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        gpu=True,
        normalize=True,
        transform_type="exponent",
        drop_last_freq=True,
        **kwargs
    ):
        super().__init__()
        self.raw_wavfile_path = raw_wavfile_path
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        if fmax is None:
            fmax = sampling_rate / 2
        self.fmax = fmax
        self.num_frames = num_frames
        self.phase_init = phase_init
        self.windows = {}
        self.num_workers = num_workers
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.drop_last_freq = drop_last_freq
        self.kwargs = kwargs

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = Specs(raw_wavfile_path=self.raw_wavfile_path,
                                   data_dir=self.train_data_dir,
                                   subset='train',
                                   shuffle_spec=True,
                                   num_frames=self.num_frames,
                                   sampling_rate=self.sampling_rate,
                                   n_fft=self.n_fft,
                                   num_mels=self.num_mels,
                                   hop_size=self.hop_size,
                                   win_size=self.win_size,
                                   fmin=self.fmin,
                                   fmax=self.fmax,
                                   format=self.format,
                                   phase_init=self.phase_init,
                                   normalize=self.normalize,
                                   drop_last_freq=self.drop_last_freq,
                                   spec_transform=self.spec_fwd,
                                   )
            self.valid_set = Specs(raw_wavfile_path=self.raw_wavfile_path,
                                   data_dir=self.val_data_dir,
                                   subset='val',
                                   shuffle_spec=False,
                                   num_frames=self.num_frames,
                                   sampling_rate=self.sampling_rate,
                                   n_fft=self.n_fft,
                                   num_mels=self.num_mels,
                                   hop_size=self.hop_size,
                                   win_size=self.win_size,
                                   fmin=self.fmin,
                                   fmax=self.fmax,
                                   format=self.format,
                                   phase_init=self.phase_init,
                                   normalize=self.normalize,
                                   drop_last_freq=self.drop_last_freq,
                                   spec_transform=self.spec_fwd,
                                   )
        if stage == 'test' or stage is None:
            self.test_set = Specs(raw_wavfile_path=self.raw_wavfile_path,
                                   data_dir=self.test_data_dir,
                                   subset='test',
                                   shuffle_spec=False,
                                   num_frames=self.num_frames,
                                   sampling_rate=self.sampling_rate,
                                   n_fft=self.n_fft,
                                   num_mels=self.num_mels,
                                   hop_size=self.hop_size,
                                   win_size=self.win_size,
                                   fmin=self.fmin,
                                   fmax=self.fmax,
                                   format=self.format,
                                   phase_init=self.phase_init,
                                   normalize=self.normalize,
                                   drop_last_freq=self.drop_last_freq,
                                   spec_transform=self.spec_fwd,
                                   )

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor  # 范围压缩同时外乘一个scalar
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        return torch.stft(sig, 
                          n_fft=self.n_fft, 
                          hop_length=self.hop_size, 
                          win_length=self.win_size, 
                          window=torch.hann_window(self.win_size).to(sig.device),
                          center=True)
    
    def sig2mel(self, sig):
        out = mel_spectrogram(sig, 
                               n_fft=self.n_fft, 
                               num_mels=self.num_mels, 
                               sampling_rate=self.sampling_rate, 
                               hop_size=self.hop_size, 
                               win_size=self.win_size, 
                               fmin=self.fmin, 
                               fmax=self.fmax,
                               )
        return out

    def inv_mel(self, mel):
        out = inverse_mel(mel, 
                           n_fft=self.n_fft, 
                           num_mels=self.num_mels, 
                           sampling_rate=self.sampling_rate, 
                           hop_size=self.hop_size, 
                           win_size=self.win_size, 
                           fmin=self.fmin, 
                           fmax=self.fmax,
                           )
        out = out.abs().clamp_min_(1e-6)
        return out

    def istft(self, spec, length=None):
        return torch.istft(spec, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_size, 
                           win_length=self.win_size, 
                           window=torch.hann_window(self.win_size).to(spec.device), 
                           length=length,
                           )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
