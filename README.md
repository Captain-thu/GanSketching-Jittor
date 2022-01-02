## Sketch Your Own GAN - Jittor Reproduction
This repository is the Jittor reproduction version of [GANSketching](https://github.com/PeterWang512/GANSketching).[1]

## Getting Started

### Install depences
  ```bash
  pip install -r requirements.txt
  ```
  Meanwhile, install Jittor from `https://cg.cs.tsinghua.edu.cn/jittor/download/` following the instruction.

## Train
We provide two ways to train your model on hw_horse_riders task with/without augment.
Fisrtly, change dir to `gansketching-jittor` then
```
bash ./scripts/hw_horse_riders_with_augment.sh
```
or
```
bash ./scripts/hw_horse_riders_without_augment.sh
```
### Download Datasets and Pre-trained Models

The following scripts downloads our sketch data, our evaluation set, [LSUN](https://dl.yf.io/lsun), and pre-trained models from [StyleGAN2](https://github.com/NVlabs/stylegan2) and [PhotoSketch](https://github.com/mtli/PhotoSketch).
```bash
# Download the sketches
bash data/download_sketch_data.sh

# Download evaluation set
bash data/download_eval_data.sh

# Download pretrained models from StyleGAN2 and PhotoSketch
bash pretrained/download_pretrained_models.sh

# Download LSUN cat, horse, and church dataset
bash data/download_lsun.sh
```

To train FFHQ models with image regularization, please download the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) using this [link](https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=sharing). This is the zip file of 70,000 images at 1024x1024 resolution. Unzip the files, , rename the `images1024x1024` folder to `ffhq` and place it in `./data/image/`.




## Acknowledgments
This repository mainly construct on [GANSketching](https://github.com/PeterWang512/GANSketching)[1]

[1] Wang, S.Y., Bau, D., & Zhu, J.Y. (2021). Sketch Your Own GAN. In Proceedings of the IEEE International Conference on Computer Vision.