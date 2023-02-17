# AirLoc: Object Based Indoor Relocalisation

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](./LICENSE)
[![Air Series](https://img.shields.io/badge/collection-Air%20Series-b31b1b)](https://chenwang.site/airseries/)

## Introduction

Indoor relocalization is vital for both robotic tasks such as autonomous exploration and civil applications such as navigation with a cell phone in a shopping mall. Some previous approaches adopt geometrical information such as key-point features or local textures to carry out indoor relocalization, but they either easily fail in environments with visually similar scenes or require many database images. Inspired by the fact that humans often remember places by recognizing unique landmarks, we resort to objects, which are more informative than geometry elements. In this work, we propose a simple yet effective object-based indoor relocalization approach, dubbed AirLoc. To overcome the critical challenges including the ob- ject reidentification and remembering object relationships, we extract object-wise appearance embedding and inter-object geo- metric relationship. The geometry and appearance features are integrated to generate cumulative scene features. This results in a robust, accurate, and portable indoor relocalization system, which outperforms the state-of-the-art methods in room-level relocalization by 12% of PR-AUC and 8% of accuracy. Besides, AirLoc shows robustness in challenges like severe occlusion, perceptual aliasing, viewpoint shift, deformation, and scale transformation.


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

For Data Loading, we use dataloaders present in the datasets folder. The dataloader support preprocessed pkl files from the Reloc110 scenes.

Please download data.zip (Preprocessed Queries) and database_raw.zip (Database)

To preprocess dataset from scratch please refer to ..

## Pre-trained Models for Inference

For inference, please download the models.zip file:

* [Pre-trained Models](https://mega.nz/file/IgBVDQrD#qxdB2hNazSTbV1_QdQwO2AamWveCsBTk3AGieZ8jmDQ)

## Indoor Relocalization Evaluation

### Accuracy
Please modify the eval_Airloc.yaml config file to test for different methods and datasets.

We save the preprocessed dataset at db_path in the first run to save the computation. 
```
python eval_airloc.py -c config/eval_Airloc.yaml
```

### PR-AUC
Please modify the eval_Airloc_prauc.yaml config file to test for different methods and datasets.

```
python eval_airloc_prauc.py -c config/eval_Airloc.yaml
```

## Training
To train AirLoc Geometry Module: (Please refer to train_airloc.yaml)

```
python train/train_airloc_geometry.py -c config/train_airloc.yaml
```

To train NetVLAD: (Please refer to train_netvlad.yaml)

```
python './train/train_netvlad.py' -c './config/train_netvlad.yaml'
```


