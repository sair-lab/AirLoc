# AirLoc: Object Based Indoor Relocalisation

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](./LICENSE)
[![stars](https://img.shields.io/github/stars/Nik-V9/AirObject?style=social)](https://github.com/Nik-V9/AirObject/stargazers)
[![Air Series](https://img.shields.io/badge/collection-Air%20Series-b31b1b)](https://chenwang.site/airseries/)
[![arXiv](https://img.shields.io/badge/arXiv-2111.15150-b31b1b.svg)](https://arxiv.org/abs/2111.15150)
[![IEEE/CVF CVPR 2022](https://img.shields.io/badge/-IEEE/CVF%20CVPR%202022-blue)](https://cvpr2022.thecvf.com/)

## Introduction

Coming Soon ..


## Dependencies

Simply run the following commands:

```bash
git clone 
conda create --channel conda-forge --name airloc --file ./Airloc/conda_requirements.txt
conda activate airloc
conda install pytorch==1.8.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install pyyaml opencv-python scipy tqdm pycocotools kornia
# Editied this due to some error faced
pip install torchvision==0.9.0 tensorboard

```

## Data

For Data Loading, we use dataloaders present in the datasets folder.

Drive folder for data coming soom ..



## Pre-trained Models for Inference

For inference, please download the models.zip file:

* [Pre-trained Models](https://mega.nz/file/IgBVDQrD#qxdB2hNazSTbV1_QdQwO2AamWveCsBTk3AGieZ8jmDQ)

## SuperPoint Features Extraction

We first start by pre-extracting SuperPoint features for all the images. Please modify the `superpoint_extraction_Airloc.yaml` config file to extract SuperPoint Features for different datasets:

```sh
python './Airloc/superpoint_extraction_airloc.py' -c './Airloc/config/superpoint_extraction_Airloc.yaml' -g 1
```


## Training
To preprocess the dataset:

```sh
python './datasets/preprocess_with_depth.py' -c './config/preprocess_mp3d.yaml'
```

To train Graph Attention Encoder: (Please refer to `train_gcn.yaml`)

```sh
python './train/train_airloc_v2.py' -c './config/train_airloc.yaml' -g 1
```

To train NetVLAD: (Please refer to `train_netvlad.yaml`)

```sh
python './train/train_netvlad.py' -c './config/train_netvlad.yaml' -g 1
```


To train AirObject: (Please refer to `train_airobj.yaml`)

```sh
python './train/train_airobj.py' -c './config/train_airobj.yaml' -g 1
```

## Reference Dataset Generation:
```sh
python 'generate_reference.py' -c './config/generate_reference.yaml'
```

## Evaluation:
```sh
python 'eval_airloc.py' -c './config/eval_Airloc.yaml' 
```

