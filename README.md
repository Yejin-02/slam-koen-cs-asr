<div align="center">
    <h1>
    SLAM-ASR
    </h1>
    <p>
    <b>SLAM-ASR</b> is a deep learning framework for training automatic speech recognition (ASR) models using large language models (LLMs). This repository provides a clean setup for training SLAM-ASR models on LibriSpeech dataset. <br>
    </p>
    <p>
    </p>
    <p>
    </p>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Cuda-11.8+-orange" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/PyTorch-2.01+-brightgreen" alt="python"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

# Table of Contents
1. [Usage](#usage)
2. [Features](#features)

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
