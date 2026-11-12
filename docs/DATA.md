# Data and models

## XWOD

Use the official [XWOD benchmark](https://github.com/kuanting/xwod) and its linked [dataset download](https://www.kaggle.com/datasets/kuantinglai/exwod). Keep the supplied image names and train/validation/test assignments. The dataset is described in the [XWOD paper](https://arxiv.org/abs/2605.11521).

Place the original images and YOLO annotations in this layout:

    data/xwod/
      train/images/   train/labels/
      valid/images/   valid/labels/
      test/images/    test/labels/

The on-disk validation directory is valid. Generated label and token files use the prefix val. Main PaDU evaluation uses the intersection of valid cached tokens and both label sets. The original XWOD image counts were 5945 train, 1744 validation, and 2321 test. The scored PaDU test cohort contains 2320 images.

Use the reference XWOD-tuned YOLOv8m checkpoint at runs/weatherfilm_runs/baseline_lr001/weights/best.pt. Compare its digest with [the checkpoint manifest](reference-checkpoints.json). The reference detector determines the detectability labels, so a newly trained detector can produce different labels and results. The two detector recipes are documented in [Experiment commands](REPRODUCE.md).

The label generators produce train_understanding_labels.json, train_detectability_labels.json, the corresponding val/test files, and detectability_thresholds.json under data/understanding_labels/. Keep training thresholds fixed for validation and test. The VLM extraction scripts store <split>_tokens.npy and <split>_meta.npz under data/weather_patch_tokens/. The PaDU grid is 12 by 16 with 4096 channels, stored as half-precision arrays.

## BDD-IW

Use the dataset described in [the BDD-IW paper](https://doi.org/10.3389/fnbot.2023.1058723) and the maintainer's [project and dataset information](https://github.com/ZhaoHe1023/Improved-YOLOv4). This is the BDD-IW subset used in the manuscript, so use that subset and its labels for matching results.

    data/BDD-IW/
      train/images/   train/labels/
      val/images/     val/labels/

The published protocol holds out 15% of sorted JPG names from the original training split with seed 0. Keep the original validation split. The split script creates data/bdd_understanding/splits.json, which all BDD-IW generation scripts use. The separate YOLO training view created by prep_bdd_yolo.py excludes the held-out images. Run it on a system that supports symbolic links.

## Model directories

| Model | Local path | Source |
| --- | --- | --- |
| Frozen Qwen3.5-9B | models/Qwen3.5-9B/ | [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Reference XWOD YOLOv8m | runs/weatherfilm_runs/baseline_lr001/weights/best.pt | Reference checkpoint fingerprint in this repository. Obtain a matching copy or rerun detector training as described in the experiment guide. |
| BDD-IW YOLOv8m | runs/weatherfilm_runs/bdd_baseline-2/weights/best.pt | Train on the carved BDD-IW training split or use the corresponding author-provided checkpoint. |
| COCO RT-DETRv2-R50 | models/rtdetrv2_r50_coco/ | [Pinned upstream revision](https://huggingface.co/PekingU/rtdetr_v2_r50vd/tree/282494075698cab9faa1096ae26856890030c817) |

Download the complete Qwen snapshot locally. The extraction scripts read the local model files. QWEN_PATH can point text-generation scripts to the same snapshot. RT-DETRv2-R50 needs config.json, preprocessor_config.json, and model.safetensors. Its model file SHA-256 is 3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71.

Large inputs, caches, and outputs are ignored by Git. Provide data access through upstream sources and, where necessary, an authorized location for exact reference artifacts.

## Evaluation protocol

Results use the image-level partitions described above. XWOD has known exact duplicates across splits. Source-scene independence has not been established for these partitions. Report the original split results and duplicate-exclusion sensitivity analysis separately.

The risk evaluation measures offline detector performance in predicted risk groups. Runtime measurements use an RTX 4090 with batch size one.
