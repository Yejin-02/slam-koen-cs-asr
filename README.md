<div align="center">
    <h1>
    SLAM-ASR
    </h1>
    <p>
    <b>SLAM-ASR</b> is a deep learning framework for training automatic speech recognition (ASR) models using large language models (LLMs). This repository provides a clean setup for training SLAM-ASR models on LibriSpeech dataset. <br>
    </p>
    <p>
    <img src="docs/logo.jpg" alt="SLAM-LLM Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Cuda-11.8+-orange" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/PyTorch-2.01+-brightgreen" alt="python"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

# Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Citation](#citation)

# Installation
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.35.2
pip install -e .
cd ..
git clone https://github.com/huggingface/peft.git
cd peft
git checkout tags/v0.6.0
pip install -e .
cd ..
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/ddlBoJack/SLAM-LLM.git
cd SLAM-LLM
pip install  -e .
```

For Hubert encoder, you may need to use `fairseq`, the command line is as follows:
```
# you need to install fairseq before SLAM-LLM
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

We also provide a docker image for convenience:
```shell
# build docker image
docker build -t slam-llm:latest .

# run docker image with gpu
docker run -it --gpus all --name slam --shm-size=256g slam-llm:latest /bin/bash
```

# Usage

Please refer to [examples/asr_librispeech/README.md](examples/asr_librispeech/README.md) for detailed usage instructions.

## Supported Encoders
- **Whisper**: OpenAI's Whisper encoder (whisper-large-v3-turbo)
- **WavLM**: WavLM-Large encoder
- **Hubert**: Hubert XtraLarge encoder

## Configuration Priority
We provide hierarchical configuration inheritance relationships as follows:
```
command-line (shell file) > Hydra configuration (yaml file) > dataclass configuration (Python file)
```

# Features
- Easily extend to new models and tasks.
- Detailed recipes for training and high-performance checkpoints for inference.
- Mixed precision training which trains faster with less GPU memory on NVIDIA tensor cores. 
- Multi-GPU training with data and model parallel, supporting [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [deepspeed](https://github.com/microsoft/DeepSpeed).  
- Flexible configuration based on [Hydra](https://github.com/facebookresearch/hydra) and [dataclass](https://docs.python.org/3/library/dataclasses.html) allowing a combination of code, command-line and file based configuration. 

# Citation

SLAM-ASR:
```
@article{ma2025speech,
  title={Speech Recognition Meets Large Language Model: Benchmarking, Models, and Exploration},
  author={Ma, Ziyang and Yang, Guanrou and Yang, Yifan and Gao, Zhifu and Wang, Jiaming and Du, Zhihao and Yu, Fan and Chen, Qian and Zheng, Siqi and Zhang, Shiliang and others},
  journal={Proc. AAAI},
  year={2025}
}
```
