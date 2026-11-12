# PaDU: Perception-to-Detectability Understanding

PaDU predicts object-detector reliability in extreme-weather driving scenes. It reads intermediate visual patch tokens from a frozen vision–language model and trains a lightweight decoder for weather severity, object density and composition, and detector-specific detectability. This repository contains the centralized and federated implementations and the evaluation protocols used in the manuscript.

## Repository layout

| Path | Purpose |
| --- | --- |
| [core/](core/) | Patch-token extraction, programmatic labels, the PaDU decoder, centralized training, and representation baselines |
| [fed_padu.py](fed_padu.py), [gpu_federation/](gpu_federation/) | Federated training, participation, quantized uplinks, aggregation, and checkpointed runs |
| [gpu_representation/](gpu_representation/) | Targeted text baseline and layer-ablation configurations |
| [experiments/](experiments/) | Statistical-feature failure prediction and RT-DETRv2-R50 training and evaluation |
| [configs/](configs/) | Per-seed experiment configurations and the recorded software environment |
| [docs/](docs/) | Data setup, experiment commands, and reference checkpoint fingerprints |

## Setup

Use Python 3.12 on a Linux machine with a CUDA-capable GPU. The original training and timing experiments used an RTX 4090 with 24 GB VRAM. The frozen model and feature caches require additional storage on the machine running the experiments.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    # Install a matching CUDA PyTorch and torchvision pair for your system.
    # Install a Transformers build with Qwen3.5 support.
    python -m pip install -r requirements.txt

The experiments used PyTorch 2.12.0+cu126, torchvision 0.27.0+cu126, and Transformers 5.8.0.dev0. Package versions are listed in [the environment record](configs/experiment-environment.json). Install PyTorch for the CUDA version on your machine using the [official selector](https://pytorch.org/get-started/locally/). The environment record identifies the Transformers development version but does not pin its source commit.

Run commands from the repository root. Set cache locations on a disk with sufficient capacity before installing or running models if you do not want these files on the system disk.

    export HF_HOME="$PWD/models/hf-cache"
    export TORCH_HOME="$PWD/models/torch-cache"
    export QWEN_PATH="models/Qwen3.5-9B"

## Data and model setup

Acquire XWOD and BDD-IW from their maintainers. See [Data and models](docs/DATA.md) for official links, folder layouts, the split protocol, and required weights. Images, detector weights, PaDU checkpoints, and cached activations are acquired or generated separately. The [reference checkpoint manifest](docs/reference-checkpoints.json) identifies the weights used for comparisons by SHA-256.

## Reproduce the main XWOD experiment

After preparing the input data and reference detector according to [Data and models](docs/DATA.md):

    python core/gen_understanding_labels.py --split all
    python core/gen_detectability_labels.py --split all
    python core/extract_patch_tokens.py --layer -4

    for seed in 0 1 2; do
      python core/train_understanding.py --seed "$seed" --name uwm_main --lam 3
    done

    python core/eval_test.py --name uwm_main --seeds 0 1 2
    python core/eval_per_weather.py --name uwm_main --seeds 0 1 2
    python export_predictions.py
    python summarize_metrics.py

PaDU checkpoints are written to runs/weatherfilm_runs/uwm_main_seed*/best_uwm.pt. Risk scores and test summaries are written under runs/A1_A2/, runs/A1/, and runs/A2/.

## Reproduce the comparisons

See [Experiment commands](docs/REPRODUCE.md) for the representation baselines, BDD-IW, layer ablation, targeted prompts, statistical features, cross-detector transfer, warm latency, and federated scenarios. For example, preview a configuration without loading the data or starting training:

    python train.py --config configs/B3_weather_seed0.json --dry-run

## Citation

See [CITATION.cff](CITATION.cff) for the manuscript title and authors.

## Acknowledgements

This implementation uses PyTorch, Hugging Face Transformers, timm, and Ultralytics. Upstream software, models, and datasets are listed in [Third-party resources](docs/THIRD_PARTY.md).
