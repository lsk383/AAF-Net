# AAF-Net

Official PyTorch implementation of **AAF-Net: Adaptive Attention Fusion Network for Bi-temporal Landslide Change Detection**.

## Introduction

AAF-Net is designed for detecting newly occurring landslides from pre-event and post-event remote-sensing images.

The network mainly addresses two challenges: weak change responses caused by the spectral similarity between pre-event bare surfaces and post-event landslides, and incomplete predictions caused by irregular landslide boundaries and complex internal textures.

AAF-Net contains two main components:

* **Triplet Attention Fusion Module (TAFM):** jointly models pre-event features, post-event features, and their absolute difference features, and adaptively recalibrates the fused change features.
* **Hierarchical Attention-Guided Decoder (HAGD):** uses deep semantic information to progressively guide shallow spatial-detail recovery, improving the continuity and integrity of landslide regions.

## Repository Structure

```text
AAF-Net/
├── models/
│   ├── __init__.py
│   ├── LWGANet.py
│   └── model_with_tafm.py
├── Transforms.py
├── dataset_GVLM.py
├── metric_tool.py
├── utils.py
├── train_aafnet.py
├── test_aafnet.py
├── requirements.txt
└── README.md
```

## Requirements

The code was tested with Python 3.8 and PyTorch 1.12.1.

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Dataset

### GVLM-CD

The GVLM-CD dataset can be downloaded from:

* [Download GVLM-CD](YOUR_GVLM_CD_DATASET_LINK)

After downloading and extracting the dataset, specify its root directory through the `--file_root` argument.

Example dataset path:

```text
/path/to/GVLM-CD-Processed
```

## Pretrained Weights

The best-performing AAF-Net checkpoint trained on GVLM-CD is available here:

* [Download best_model.pth](https://github.com/lsk383/AAF-Net/releases/latest/download/best_model.pth)

The released checkpoint corresponds to an input size of 256 × 256 pixels.

## Training

Run the following command to train AAF-Net:

```bash
python train_aafnet.py \
    --file_root /path/to/GVLM-CD-Processed
```

Default training settings:

* Input size: 256 × 256
* Batch size: 32
* Initial learning rate: 5e-4
* Maximum training steps: 20,000
* Learning-rate policy: polynomial decay

## Testing

Download `best_model.pth` and run:

```bash
python test_aafnet.py \
    --file_root /path/to/GVLM-CD-Processed \
    --weight /path/to/best_model.pth
```

Replace the dataset and weight paths according to the local environment.

## Results

Quantitative results on the GVLM-CD test set:

| Method  | F1-Score (%) | IoU (%) | Precision (%) | Recall (%) | FLOPs (G) |
| ------- | -----------: | ------: | ------------: | ---------: | --------: |
| AAF-Net |        92.11 |   85.38 |         90.92 |      93.33 |      5.99 |

The model contains approximately **12.91 million parameters**. FLOPs are calculated with two 256 × 256 RGB images as input.

## Acknowledgements

The backbone network is based on LWGANet. We thank the authors of the related open-source projects and datasets for their contributions.
# AAF-Net
Adaptive Attention Fusion Network for bi-temporal landslide change detection.
