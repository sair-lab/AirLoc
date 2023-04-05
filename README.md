# AirLoc: Object Based Indoor Relocalisation

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](./LICENSE)
[![Air Series](https://img.shields.io/badge/collection-Air%20Series-b31b1b)](https://chenwang.site/airseries/)

## Introduction

Indoor relocalization is vital for both robotic tasks such as autonomous exploration and civil applications such as navigation with a cell phone in a shopping mall. Some previous approaches adopt geometrical information such as key-point features or local textures to carry out indoor relocalization, but they either easily fail in environments with visually similar scenes or require many database images. Inspired by the fact that humans often remember places by recognizing unique landmarks, we resort to objects, which are more informative than geometry elements. In this work, we propose a simple yet effective object-based indoor relocalization approach, dubbed AirLoc. To overcome the critical challenges including the object reidentification and remembering object relationships, we extract object-wise appearance embedding and inter-object geometric relationship. The geometry and appearance features are integrated to generate cumulative scene features. This results in a robust, accurate, and portable indoor relocalization system, which outperforms the state-of-the-art methods in room-level relocalization by 12% of PR-AUC and 8% of accuracy. Besides, AirLoc shows robustness in challenges like severe occlusion, perceptual aliasing, viewpoint shift, deformation, and scale transformation.


## Live relocalization demo

![AirLoc](https://user-images.githubusercontent.com/8695500/227297907-6dfc6219-b5f2-48b9-bd31-6ae48de0a476.png)

## Dependencies

Simply run the following commands:

```bash
git clone https://github.com/sair-lab/AirLoc.git
conda create --channel conda-forge --name airloc --file ./AirLoc/conda_requirements.txt
conda activate airloc
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pyyaml opencv-python scipy tqdm tensorboard
pip install kornia==0.5.0


```

## Data

For Data Loading, we use dataloaders present in the datasets folder. The dataloader support preprocessed pkl files from the Reloc110 scenes.

Please download data.zip (Preprocessed Queries) and database_raw.zip (Database)
* [Dataset (Small Version)](https://drive.google.com/drive/folders/1n2wz_bigMcM5l9K29bskMd1qhR2VLarH?usp=sharing)
* [Dataset (Full Version)](https://entuedu-my.sharepoint.com/:f:/g/personal/cwang017_e_ntu_edu_sg/EqkVfge8ADpHg2uf9jJexOsBW6SZ6vpyMf-lM59vZj1xHg?e=B4yvs0)

Note: Data preprocessing is not required if you download preprocessed dataset from above link. But to preprocess dataset from scratch please refer to [Preprocessing](https://github.com/aryanmangal769/AirLoc-Object-Based-Inddor-Relocalization/blob/main/datasets/readme.md)
 

The expected directory structure after preprocessing (or directly downloading preprocessed data):

```
data_collection/
   data/
      RPmz2sHmrrY.pkl
      S9hNv5qa7GM.pkl
            .
            .
   database_raw/
      mp3d/
         RPmz2sHmrrY/
               rooms/
         S9hNv5qa7GM/
              .
              .     
```

## Pre-trained Models for Inference

For inference, please download the models.zip file:

* [Pre-trained Models](https://drive.google.com/drive/folders/1n2wz_bigMcM5l9K29bskMd1qhR2VLarH?usp=sharing)

Expected directory structure:

```
\models
   netvlad_model.pth
   gcn_model.pth
         .
```        

## Indoor Relocalization Evaluation

### Accuracy
Please modify the eval_Airloc.yaml config file to test for different methods and datasets.
* method: The method you want to generate test results
* scenes: The scneces(Reloc110) you want to get test results for

* base_dir: path to data folder
* db_raw_path: path to database_raw folder
* db_path: empty folder for saving preprocessed database

We save the preprocessed dataset at db_path in the first run to save time in further runs. 
```
python eval_airloc.py -c config/eval_Airloc.yaml
```

### PR-AUC
Please modify the eval_Airloc_prauc.yaml config file to test for different methods and datasets.

```
python eval_airloc_prauc.py -c config/eval_airloc_prauc.yaml

```

## Training
To train AirLoc Geometry Module: (Please refer to train_airloc.yaml)

```
python train/train_airloc_geometry.py -c config/train_airloc.yaml
```

# Watch Video

[<img src="https://user-images.githubusercontent.com/8695500/227299683-3abd8f8e-636f-416b-bf7f-1d716f53991e.png" width="80%">](https://youtu.be/n6mp6KzGPgs)

# Puplication
```bibtex
@article{aryan2023airloc,
  title = {AirLoc: Object-based Indoor Relocalization},
  author = {Aryan and Li, Bowen and Scherer, Sebastian and Lin, Yun-Jou and Wang, Chen},
  journal = {arXiv preprint arXiv:2304.00954},
  year = {2023},
}
```
You may also [download this paper](https://arxiv.org/abs/2304.00954).
