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
# Score also means bridge model here
from div.model import ScoreModelGAN
from div.util.other import pad_spec
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
    ckpt_path = exp_path + '/mode=bridge-only_sde=BridgeGAN_backbone=ncspp_l_crm_sde_type_gmax_c_0.4_k_2.6_beta_max_20.0score_mse1.0,multi-mel0.1_mel100.ckpt'

    parser = ArgumentParser()
    parser.add_argument("--script_dir", type=str, required=True, default="/path/", 
                        help='Directory containing the test data')
    parser.add_argument("--test_scp_dir", type=str, required=False, default="./Datascp/ref_audio/", 
                        help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=False, default=enh_dir, 
                        help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str, required=False, default=ckpt_path,
                        help='Path to model checkpoint')
    parser.add_argument("--sample_steps", type=int, default=10, help="Number of reverse steps")

    exp_name = ckpt_path.split("/")[-1]
    mode = exp_name.split("_")[0].split("=")[1]
    sde = exp_name.split("_")[1].split("=")[1].lower()
    backbone = 'ncspp_l_crm'
    if sde=='bridgegan':
        beta_max = exp_name.split("_beta_max_")[1].split('_')[0]
        print('beta_max_from_file=',beta_max)
    elif sde=='ouvesde':
        sde = "ouve"
        parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")


    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls_score = BackboneRegistry.get_by_name(backbone) if backbone != "none" else None

    sde_class = SDERegistry.get_by_name(sde)
    model_cls = ScoreModelGAN
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
    score_model = ScoreModelGAN(backbone=backbone, sde=sde, data_module_cls=data_module_cls,
                **{
                    **vars(arg_groups['ScoreModelGAN']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['BackboneScore']),
                    **vars(arg_groups['DataModule'])
                })
    model = score_model.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.dnn.to(args.device)
    model.eval()

    model.sde.N = args.N
    model.sde.sampling_type = args.sampling_type
    model.num_mels = args.num_mels
    print('N=', model.sde.N, model.sde.sampling_type)

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
        norm_factor = torch.max(torch.abs(data)) + 1e-6
        data = data / norm_factor
        
        # Prepare DNN input
        Y = model._forward_transform(data).unsqueeze(1)  # (1, 1, F, T)
        Y = pad_spec(Y)

        # add phase
        phase_ = torch.zeros_like(Y).to(Y.device)
        Y = torch.complex(Y * torch.cos(phase_), Y * torch.sin(phase_))  # complex-tensor, (B, 1, F, T)
        Y = Y[:, :, :-1].contiguous()

        # range-adjust
        Y = model.data_module.spec_fwd(Y)
        Y = torch.cat([Y.real, Y.imag], dim=1)

        if sde.lower() == 'ouve':
            # Reverse sampling
            sampler = model.get_pc_sampler(
                'reverse_diffusion', args.corrector, Y.to(args.device), N=args.sample_steps, 
                corrector_steps=args.corrector_steps, snr=args.snr)
            sample, _ = sampler()  # (B, 2, F-1, T)
        elif 'bridgegan' in sde.lower():
            sample = model.sde.reverse_diffusion(Y.to(args.device), Y.to(args.device), model.dnn)  # (B,2,F-1,T)

        sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # (B,1,F-1,T)
        sample_last = sample[:, :, -1].unsqueeze(-2).contiguous()  # (B, 1, 1, T)
        sample = torch.cat([sample, sample_last], dim=-2)  # (B, 1, F, T)

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor.cpu()

        # Write enhanced wav file
        makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), args.sampling_rate)