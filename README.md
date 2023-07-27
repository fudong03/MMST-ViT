# MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer

## Model Overview

![mmst-vit-arch](./input/mmst-vit-arch.png)



This repository provides the official implementation of our proposed Multi-Modal Spatial-Temporal Vision Transformer (MMST-ViT), developed for predicting crop yields at the county level across the United States. It consists of a Multi-Modal Transformer, a Spatial Transformer, and a Temporal Transformer. The Multi-Modal Transformer leverages satellite images and meteorological data during the growing season for capturing the direct impact of short-term weather variations on crop growth. The Spatial Transformer learns the high-resolution spatial dependency among counties for precise crop tracking. The Temporal Transformer captures the effects of long-term climate change on crops.



## Requirements

Our model is based on the following libraries:

- torch
- torchvision
- timm
- numpy
- pandas
- h5py
- Pillow

You can use the following instruction to install all the requirements:

```python
# install requirements
pip install -r requirements.txt
```



## Pre-training

![method-pvt-simclr](./input/method-pvt-simclr.png)



The above figure illustrates the architexture of our proposed multi-modal self-supervised pre-training.

 To pre-train MMST-ViT, please run the following commend:

```python
# pre-train
python main_pretrain.py
```



## Fine-tuning

To fine-tune MMST-ViT for crop yield predictions, use the following commend:

```python
# fine-tune
python main_finetune_mmst_vit.py
```

