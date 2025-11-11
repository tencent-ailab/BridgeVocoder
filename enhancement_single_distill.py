# Here is the script for the BAC model after distillation inference.
# nmel=100,fmax=12000,sr=24000.
import os
import librosa as lib
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from os.path import join, dirname
from argparse import ArgumentParser
from div.data_module import SpecsDataModule
from div.backbones.shared import BackboneRegistry
from div.sdes import SDERegistry
from div.model import SinModel
import argparse

def get_argparse_groups(parser):
	groups = {}
	for group in parser._action_groups:
		group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions}
		groups[group.title] = argparse.Namespace(**group_dict)
	return groups

if __name__ == '__main__':
    exp_root = './ckpt/'
    exp_path = exp_root + 'LibriTTS'
    enh_dir = exp_path + '/enh/'
    # put the ckpt here
    ckpt_path = exp_path + '/xxx.ckpt'

    parser = ArgumentParser()
    parser.add_argument("--script_dir", type=str, required=True, default="/path/", 
                        help='Directory containing the test data')
    parser.add_argument("--test_scp_dir", type=str, required=False, default="./Datascp/ref_audio/", 
                        help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=False, default=enh_dir, 
                        help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str, required=False, default=ckpt_path,
                        help='Path to model checkpoint')

    exp_name = ckpt_path.split("/")[-1]
    mode = exp_name.split("_")[0].split("=")[1]
    sde = exp_name.split("_")[1].split("=")[1].lower()
    backbone = 'bac'
    beta_max = exp_name.split("_beta_max_")[1].split('_')[0]
    print('beta_max_from_file=',beta_max)

    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls_score = BackboneRegistry.get_by_name(backbone) if backbone != "none" else None

    sde_class = SDERegistry.get_by_name(sde)
    model_cls = SinModel
    model_cls.add_argparse_args(
        parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))
    sde_class.add_argparse_args(
        parser.add_argument_group("SDE", description=sde_class.__name__))

    backbone_cls_score.add_argparse_args(
        parser.add_argument_group("BackboneScore", description=backbone_cls_score.__name__))

    # Add data module args
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__))
    
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    # Load score model 
    score_model = SinModel(backbone=backbone, sde=sde, data_module_cls=data_module_cls,
                **{
                    **vars(arg_groups['ScoreModelGAN']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['BackboneScore']),
                    **vars(arg_groups['DataModule'])
                })
    model = score_model.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.dnn.to(args.device)
    model.eval()

    model.sde.sampling_type = args.sampling_type
    model.num_mels = args.num_mels

    # Get list of noisy files
    post_str = os.path.splitext(args.test_scp_dir)[-1]
    if post_str in ['.txt', '.scp']:
        filelist = []
        lines = open(args.test_scp_dir, 'r').readlines()
        for l in lines:
            cur_filename = '/'.join((l.strip().split('|')[0]).split('/')[1:3])
            filelist.append(os.path.join(args.raw_wavfile_path, cur_filename))
    else:  # dir
        filelist = sorted([args.test_scp_dir + f for f in os.listdir(args.test_scp_dir) if f.endswith('.wav')])


    # Enhance files
    for noisy_file in tqdm(filelist):
        filename = os.path.split(noisy_file)[-1]
        data, _ = lib.load(noisy_file, sr=args.sampling_rate, mono=True)
        data = torch.FloatTensor(data.astype('float32')).unsqueeze(0).to(args.device)  # （1, L)
        T_orig = data.shape[-1]

        # Normalize
        if args.normalize:
            norm_factor = torch.max(torch.abs(data)) + 1e-6
        else:
            norm_factor = 1.0

        data = data / norm_factor
        
        # Prepare DNN input
        Y = model._forward_transform(data).unsqueeze(1)  # (1, 1, F, T)

        # add phase
        phase_ = torch.zeros_like(Y).to(Y.device)
        Y = torch.complex(Y * torch.cos(phase_), Y * torch.sin(phase_))  # complex-tensor, (B, 1, F, T)
        if args.drop_last_freq:
            Y = Y[:, :, :-1].contiguous()

        # range-adjust
        Y = model.data_module.spec_fwd(Y)
        Y = torch.cat([Y.real, Y.imag], dim=1)

        t = (torch.ones([Y.shape[0]]) * (1 - model.sde.offset)).to(Y.device)
        sample = model.dnn(inpt=Y, cond=Y, time_cond=t)

        sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # (B,1,F-1,T)
        if args.drop_last_freq:
            sample_last = sample[:, :, -1].unsqueeze(-2).contiguous()  # (B, 1, 1, T)
            sample = torch.cat([sample, sample_last], dim=-2)  # (B, 1, F, T)

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor.cpu()

        # Write enhanced wav file
        makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), args.sampling_rate)