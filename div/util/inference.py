import math
import torch
from div.util.other import pad_spec
from div.util.periodicity import calculate_periodicity_metrics
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
import librosa as lib

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_module import load_wav


def evaluate_score_model(model, num_eval_files):
    model.eval()
    wav_files = model.data_module.valid_set.wav_files
    # Select test files uniformly accros validation files
    total_num_files = len(wav_files)
    indices = torch.linspace(0, total_num_files - 1, num_eval_files, dtype=torch.int)
    wav_files = list(wav_files[i] for i in indices)

    _pesq = 0
    _estoi = 0
    _periodicity = 0
    count = 0
    # iterate over files
    for wav_file in tqdm(wav_files):
        # Load wavs
        audio = load_wav(wav_file, model.data_module.valid_set.sampling_rate)
        audio = torch.FloatTensor(audio).unsqueeze(0)  # (1, L)
        T_orig = audio.size(1)   

        # Normalize per utterance
        if model.data_module.normalize:
            norm_factor = torch.max(torch.abs(audio)) + 1e-6
        else:
            norm_factor = 1.0

        audio = audio / norm_factor

        # Prepare DNN input
        Y = model._forward_transform(audio).unsqueeze(1)  # (1, 1, F, T)
        Y = pad_spec(Y)
        # add phase
        if model.data_module.valid_set.phase_init == "random":
            phase_ = 2 * math.pi * torch.rand_like(Y) - math.pi
        elif model.data_module.valid_set.phase_init == "zero":
            phase_ = torch.zeros_like(Y)
        Y = torch.complex(Y * torch.cos(phase_), Y * torch.sin(phase_))  # (B, 1, F, T)
        # whether drop the last frequency
        if model.data_module.valid_set.drop_last_freq:
            Y = Y[:, :, :-1].contiguous()
        
        # value range adjust
        Y = model.data_module.spec_fwd(Y)
        
        audio = audio * norm_factor
        
        Y_ = torch.cat([Y.real, Y.imag], dim=1)
        Y_ = Y_.cuda()
        cond = Y_
        if "bridge" in model.sde_name.lower():
            # for IRSDE reverse
            sample = model.sde.reverse_diffusion(Y_, cond, model.dnn)
            sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # complex-tensor, (B,1,F-1,T)
        elif model.sde_name.lower() == "ouve":#print(model.sde.reverse_n,'!!!')
            sampler = model.get_pc_sampler('reverse_diffusion', 'ald', Y_.cuda(), N=model.sde.reverse_n,
                                           corrector_steps=model.sde.corrector_steps,
                                           snr=model.sde.snr)
            sample, _ = sampler()  # (B, 2, F-1, T)
            sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # (B, 1, F-1, T)
        elif model.sde_name.lower() in ["rtf", "otf"]:  # Rectifiled Flow-Matching
            z0 = torch.randn(*Y_.shape, dtype=Y_.dtype, device=Y_.device)
            sample = model.sde.reverse_diffusion(z0, cond, model.dnn)
            sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # complex-tensor, (B, 1, F-1, T)

        if model.data_module.valid_set.drop_last_freq:
            sample_last = sample[:, :, -1].unsqueeze(-2).contiguous()  # (B, 1, 1, T)
            #print(sample_last.shape,sample.shape)
            sample = torch.cat([sample, sample_last], dim=-2)  # (B, 1, F, T)

        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().contiguous().cpu().numpy()
        audio = audio.squeeze().contiguous().cpu().numpy()

        audio_16k = lib.core.resample(audio, orig_sr=model.kwargs['sampling_rate'], target_sr=16000)
        x_hat_16k = lib.core.resample(x_hat, orig_sr=model.kwargs['sampling_rate'], target_sr=16000)
        audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
        x_hat_16k_tensor = torch.from_numpy(x_hat_16k).unsqueeze(0)
        #print(audio_16k.shape, x_hat_16k.shape)
        try:
            _pesq += pesq(16000, audio_16k, x_hat_16k, 'wb') 
            _estoi += stoi(audio, x_hat, model.kwargs['sampling_rate'], extended=True)
            periodicity_loss = calculate_periodicity_metrics(audio_16k_tensor, x_hat_16k_tensor)[0]
            _periodicity += periodicity_loss
            count += 1
        except:
            _pesq += 0
            _estoi += 0
            _periodicity += 0

    if count == 0:
        raise NotImplementedError("Track invalid inference result, please check the inference procedure carefully!")        
    
    return _pesq / count, _estoi / count, _periodicity / count

def evaluate_sin_model(model, num_eval_files):
    model.eval()
    wav_files = model.data_module.valid_set.wav_files
    # Select test files uniformly accros validation files
    total_num_files = len(wav_files)
    indices = torch.linspace(0, total_num_files - 1, num_eval_files, dtype=torch.int)
    wav_files = list(wav_files[i] for i in indices)

    _pesq = 0
    _estoi = 0
    _periodicity = 0
    count = 0
    # iterate over files
    for wav_file in tqdm(wav_files):
        # Load wavs
        audio = load_wav(wav_file, model.data_module.valid_set.sampling_rate)
        audio = torch.FloatTensor(audio).unsqueeze(0)  # (1, L)
        T_orig = audio.size(1)   

        # Normalize per utterance
        if model.data_module.normalize:
            norm_factor = torch.max(torch.abs(audio)) + 1e-6
        else:
            norm_factor = 1.0
        audio = audio / norm_factor

        # Prepare DNN input
        Y = model._forward_transform(audio).unsqueeze(1)  # (1, 1, F, T)
        Y = pad_spec(Y)
        # add phase
        if model.data_module.valid_set.phase_init == "random":
            phase_ = 2 * math.pi * torch.rand_like(Y) - math.pi
        elif model.data_module.valid_set.phase_init == "zero":
            phase_ = torch.zeros_like(Y)
        Y = torch.complex(Y * torch.cos(phase_), Y * torch.sin(phase_))  # (B, 1, F, T)
        # whether drop the last frequency
        if model.data_module.valid_set.drop_last_freq:
            Y = Y[:, :, :-1].contiguous()
        
        # value range adjust
        Y = model.data_module.spec_fwd(Y)

        audio = audio * norm_factor
        
        Y_ = torch.cat([Y.real, Y.imag], dim=1)
        Y_ = Y_.cuda()
        cond = Y_
        # single-step inference
        t = (torch.ones([Y_.shape[0]]) * (1 - model.sde.offset)).to(Y_.device)
        sample = model.dnn(inpt=Y_, cond=cond, time_cond=t)  # (B, 2, F, T)
        sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # complex-tensor

        if model.data_module.valid_set.drop_last_freq:
            sample_last = sample[:, :, -1].unsqueeze(-2).contiguous()  # (B, 1, 1, T)
            sample = torch.cat([sample, sample_last], dim=-2)  # (B, 1, F, T)

        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().contiguous().cpu().numpy()
        audio = audio.squeeze().contiguous().cpu().numpy()

        try:
            audio_16k = lib.core.resample(audio, orig_sr=model.kwargs['sampling_rate'], target_sr=16000)
            x_hat_16k = lib.core.resample(x_hat, orig_sr=model.kwargs['sampling_rate'], target_sr=16000)
            audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
            x_hat_16k_tensor = torch.from_numpy(x_hat_16k).unsqueeze(0)
           
            _pesq += pesq(16000, audio_16k, x_hat_16k, 'wb') 
            _estoi += stoi(audio, x_hat, model.kwargs['sampling_rate'], extended=True)
            periodicity_loss = calculate_periodicity_metrics(audio_16k_tensor, x_hat_16k_tensor)[0]
            _periodicity += periodicity_loss
            count += 1 
        except:
            _pesq += 0
            _estoi += 0
            _periodicity += 0

    if count == 0:
        raise NotImplementedError("Track invalid inference result, please check the inference procedure carefully!")

    return _pesq / count, _estoi / count, _periodicity / count
