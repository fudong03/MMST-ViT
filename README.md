# MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer

## Model Overview

![mmst-vit-arch](./input/mmst-vit-arch.png)



This repository provides the official implementation of our proposed Multi-Modal Spatial-Temporal Vision Transformer (MMST-ViT), developed for predicting crop yields at the county level across the United States. It consists of a Multi-Modal Transformer, a Spatial Transformer, and a Temporal Transformer. The Multi-Modal Transformer leverages satellite images and meteorological data during the growing season to capture the direct impact of short-term weather variations on crop growth. The Spatial Transformer learns the high-resolution spatial dependency among counties for precise crop tracking. The Temporal Transformer captures the effects of long-term climate change on crops.



## Requirements

Our model is based on the following libraries:

- torch == 1.13.0
- torchvision == 0.14.0
- timm == 0.5.4
- numpy == 1.24.4
- pandas == 2.0.3
- h5py == 3.9.0
- einops == 0.6.1
- Pillow == 10.0.0
- argparse == 1.4.0
- tqdm == 4.65.0
- scikit-learn == 1.3.0
- tensorboard == 2.13.0

You can use the following instructions to install all the requirements:

```python
# install requirements
pip install -r requirements.txt
```



## Pre-training

![method-pvt-simclr](./input/method-pvt-simclr.png)



The above figure illustrates the architecture of our proposed multi-modal self-supervised pre-training.

 To pre-train MMST-ViT, please run the following command:

```python
# pre-train
python main_pretrain_mmst_vit.py
```



## Fine-tuning

To fine-tune MMST-ViT for crop yield predictions, use the following command:

```python
# fine-tune
python main_finetune_mmst_vit.py
```

## License

This repository is under the CC-BY-NC 4.0 license. Please refer to [LICENSE](https://github.com/fudong03/MMST-ViT/blob/main/LICENSE) for details.

## Acknowledgment

This repository is based on the official implementation of [PVT](https://github.com/whai362/PVT) and [MAE](https://github.com/facebookresearch/mae). We thank the authors for releasing the code.

## Citation

```
@inproceedings{fudong:iccv23:mmst_vit,
  title={MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer},
  author={Fudong Lin and Summer Crawford and Kaleb Guillot and Yihe Zhang and Yan Chen and Xu Yuan and Li Chen and Shelby Williams and Robert Minvielle and Xiangming Xiao and Drew Gholson and Nicolas Ashwell and Tri Setiyono and Brenda S. Tubana and Lu Peng and Magdy Bayoumi and Nian-Feng Tzeng},
  booktitle={2023 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2023, Paris, France, October 2-6, 2023},
  year={2023}
}
```
