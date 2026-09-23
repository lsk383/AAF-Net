# AAF-Net

Official PyTorch implementation of:

**AAF-Net: An Adaptive Attention Fusion Network for Landslide Change Detection**

Published in *IEEE Geoscience and Remote Sensing Letters*, 2026.

* [IEEE Xplore](https://ieeexplore.ieee.org/document/11476940)
* [DOI](https://doi.org/10.1109/LGRS.2026.3681575)

## Introduction

AAF-Net is designed to detect newly occurring landslides from pre-event and post-event remote-sensing images.

The network mainly addresses two challenges in landslide change detection: weak change responses caused by the spectral similarity between pre-event bare surfaces and post-event landslides, and incomplete predictions caused by irregular landslide boundaries and complex internal textures.

AAF-Net contains two main components:

* **Triplet Attention Fusion Module (TAFM):** jointly models pre-event features, post-event features, and their absolute difference features, and adaptively recalibrates the fused change features.
* **Hierarchical Attention-Guided Decoder (HAGD):** uses deep semantic information to progressively guide shallow spatial-detail recovery, improving the continuity and integrity of detected landslide regions.

## Repository Structure

```text
AAF-Net/
├── models/
│   ├── __init__.py
│   ├── LWGANet.py
│   └── model_with_tafm.py
├── .gitignore
├── README.md
├── requirements.txt
├── Transforms.py
├── dataset_GVLM.py
├── metric_tool.py
├── utils.py
├── train_aafnet.py
└── test_aafnet.py
```

## Requirements

The code was tested with:

* Python 3.8
* PyTorch 1.12.1
* torchvision 0.13.1

Install the required packages using:

```bash
pip install -r requirements.txt
```

Please note that `mmcv-full` may need to be installed using a wheel compatible with the local PyTorch and CUDA versions.

## Dataset

### GVLM-CD

The processed GVLM-CD dataset used in this repository can be downloaded from Baidu Netdisk:

* [Download GVLM-CD](https://pan.baidu.com/s/1QxnUkevSKep3UlQYmE_KkQ)
* Extraction code: `nc8y`

The original GVLM dataset and its corresponding publication are available here:

* [Official GVLM repository](https://github.com/zxk688/GVLM)
* [Original GVLM paper](https://doi.org/10.1016/j.isprsjprs.2023.01.018)

After downloading and extracting the dataset, specify its root directory through the `--file_root` argument.

Example dataset path:

```text
/path/to/GVLM-CD-Processed
```

## Pretrained Weights

The best-performing AAF-Net checkpoint trained on the GVLM-CD dataset is available here:

* [Download best_model.pth](https://github.com/lsk383/AAF-Net/releases/download/v1.0/best_model.pth)
* [Release page](https://github.com/lsk383/AAF-Net/releases/tag/v1.0)

The released checkpoint corresponds to an input size of 256 × 256 pixels.

## Training

Run the following command to train AAF-Net:

```bash
python train_aafnet.py \
    --file_root /path/to/GVLM-CD-Processed
```

The default training settings are:

* Input size: 256 × 256
* Batch size: 32
* Initial learning rate: 5e-4
* Maximum training steps: 20,000
* Learning-rate policy: polynomial decay

Replace `/path/to/GVLM-CD-Processed` with the actual dataset path.

## Testing

Download `best_model.pth` from the Release page and run:

```bash
python test_aafnet.py \
    --file_root /path/to/GVLM-CD-Processed \
    --weight /path/to/best_model.pth
```

Replace the dataset and weight paths according to the local environment.

## Results

Quantitative results on the GVLM-CD test set:

| Method  | F1-Score (%) | IoU (%) | Precision (%) | Recall (%) | FLOPs (G) | Parameters (M) |
| ------- | -----------: | ------: | ------------: | ---------: | --------: | -------------: |
| AAF-Net |        92.11 |   85.38 |         90.92 |      93.33 |      5.99 |          12.91 |

The computational complexity is measured using two 256 × 256 RGB images as input.

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026aafnet,
  author={Li, Shengkang and Wang, Huiquan and Cui, Peixing},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={AAF-Net: An Adaptive Attention Fusion Network for Landslide Change Detection},
  year={2026},
  volume={23},
  pages={1--5},
  doi={10.1109/LGRS.2026.3681575}
}
```

## Acknowledgements

The backbone network is based on LWGANet. We thank the authors of LWGANet, the GVLM dataset, and the related open-source projects for their contributions.
