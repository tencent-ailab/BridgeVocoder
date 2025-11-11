# BridgeVoC: Neural Vocoder with Schrödinger Bridge

<p align="left">
  <img src="./Figure 1.png" alt="Logo" width="666"/>
</p>

## 🚀 Overview

This is the official repo of BridgeVoC, which explores using the Schrödinger Bridge framework for neural vocoding. This repository contains code, training scripts, and checkpoints for experiments and audio enhancement using BridgeVoC. A related paper, "BridgeVoC: Revitalizing Neural Vocoder from a Restoration Perspective", is currently in preparation.

## 📖 Abstract

While previous diffusion-based neural vocoders typically follow a noise-to-data generation pipe-line, the linear-degradation prior of the mel-spectrogram is often neglected, resulting in limited generation quality. By revisiting the vocoding task and excavating its connection with the signal restoration task, this paper proposes a time-frequency (T-F) domain-based neural vocoder with the Schrödinger Bridge, called **BridgeVoC**, which is the first to follow the data-to-data generation paradigm. Specifically, the mel-spectrogram can be projected into the target linear-scale domain and regarded as a degraded spectral representation with a deficient rank distribution. Based on this, the Schrödinger Bridge is leveraged to establish a connection between the degraded and target data distributions. During the inference stage, starting from the degraded representation, the target spectrum can be gradually restored rather than generated from a Gaussian noise process. Quantitative experiments on LJSpeech and LibriTTS show that BridgeVoC achieves faster inference and surpasses existing diffusion-based vocoder baselines, while also matching or exceeding non-diffusion state-of-the-art methods across evaluation metrics. 

## ⚡️ Project Structure

A minimal overview of the repository layout (only the folders referenced in this README are shown):

- README.md
- Figure 1.png
- requirements.txt
- starts/
  - loss/
    - bridge_gmax_wi_multi_scale_mel.sh
- ckpt/
  - LJSpeech/
    - enh.sh
  - LibriTTS/
    - enh.sh
- (other code, configs, and scripts)

## Requirements

Install Python dependencies:

```bash
pip install -r ./requirements.txt
```

## Training

Quick start — to train the model on the LJSpeech dataset:

```bash
cd starts/loss
bash bridge_gmax_wi_multi_scale_mel.sh
```

Training script parameters

The training script supports the following key parameters (examples and defaults):

- --mode: Training mode (default: "bridge-only")
- --backbone_bridge: Backbone architecture (default: "ncspp_l_crm")
- --sde: SDE type (default: "bridgegan")
- --max_epochs: Maximum number of training epochs
- --max_steps: Maximum number of training steps
- --device: Training device (default: "cuda")

Pass these parameters to your training script or adjust the shell script to include them as needed.

## Inference

Due to storage limitations, we have saved the model checkpoints on Hugging Face at https://huggingface.co/fayelei/BridgeVoC/tree/main. You can easily download the models from Hugging Face and save them in the following directories:

- Save `ljspeech=bridge-only_sde=BridgeGAN_backbone=ncspp_l_crm_sde_type_gmax_c_0.4_k_2.6_beta_max_20.0score_mse1.0,multi-mel0.1.ckpt` in the `ckpt/LJSpeech/` folder.
- Save `libritts=bridge-only_sde=BridgeGAN_backbone=ncspp_l_crm_sde_type_gmax_c_0.4_k_2.6_beta_max_20.0score_mse1.0,multi-mel0.1_mel100.ckpt` in the `ckpt/LibriTTS/` folder.

Quick start — to enhance audio using a trained model:

For LJSpeech checkpoints:

```bash
cd ckpt/LJSpeech
bash enh.sh
```

For LibriTTS checkpoints:

```bash
cd ckpt/LibriTTS
bash enh.sh
```

## Citation

If you find this project useful in your research, please cite:

```bibtex
@inproceedings{lei2025bridgevoc,
  title={BridgeVoC: Insights into Using Schr\"odinger Bridge for Neural Vocoders},
  author={Tong Lei and Andong Li and Rilin Chen and Dong Yu and Meng Yu and Jing Lu and Chengshi Zheng},
  booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy},
  year={2025},
  url={https://openreview.net/forum?id=BygUEKotgA}
}
```

Another related paper, "BridgeVoC2: Revitalizing Neural Vocoder from a Restoration Perspective," is in preparation, which proposes a single-step scheme（https://github.com/Andong-Li-speech/BridgeVoC）.
```bibtex
@misc{li2025bridgevocrevitalizingneuralvocoder,
      title={BridgeVoC: Revitalizing Neural Vocoder from a Restoration Perspective}, 
      author={Andong Li and Tong Lei and Rilin Chen and Kai Li and Meng Yu and Xiaodong Li and Dong Yu and Chengshi Zheng},
      year={2025},
      eprint={2511.07116},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2511.07116}, 
}
```
## Acknowledgments

Some of the code in this project and the environment setup are inspired or modified from the following project:

- https://github.com/sp-uhh/sgmse

--- 

If you would like me to expand any section (examples, usage details, evaluation scripts, or a more detailed project tree), tell me which parts you want improved.
