# Third-party resources

The implementation loads the following packages through their public APIs.

| Resource | Use | Upstream source |
| --- | --- | --- |
| PyTorch and torchvision | Model training, tensor operations, and vision transforms | [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) |
| Transformers | Qwen feature extraction and RT-DETRv2 loading | [Transformers](https://github.com/huggingface/transformers) |
| timm | Frozen ImageNet ViT-B/16 baseline | [timm](https://github.com/huggingface/pytorch-image-models) |
| Ultralytics | YOLOv8m training and detector outputs | [Ultralytics](https://github.com/ultralytics/ultralytics) |

Ultralytics publishes its source under AGPL-3.0. Consult the [upstream license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for its terms.

Model and dataset download sources are documented in [Data and models](DATA.md). Each upstream project's license and model or dataset terms apply to its own materials.
