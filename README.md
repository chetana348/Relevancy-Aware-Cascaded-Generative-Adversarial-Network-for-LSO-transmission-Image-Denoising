# Relevancy-Aware-Cascaded-Generative-Adversarial-Network-for-LSO-transmission-Image-Denoising

This repository is the **official pseudocode implementation** of the paper:

> **Relevancy Aware Cascaded Generative Adversarial Network for LSO-transmission Image Denoising in CT-less PET**  
> *Authors: Chetana Krishnan and Reza Teimoorisichani 

---

## 📄 Background

LSO-based transmission imaging offers a promising path for enabling low-dose, CT-less PET acquisition. However, the inherent low photon statistics of LSO transmission images can result in high levels of noise, limiting their direct use for attenuation correction and anatomical reference.

In this work, we propose a **Relevancy Aware Cascaded GAN architecture** that leverages modified U-Net and Transformer-based hybrid generators to iteratively denoise and refine LSO-transmission images. The proposed model enhances image quality in a stepwise manner, with relevancy feedback at each stage, making it well-suited for producing high-fidelity attenuation maps and improving PET-to-PET/CT translation. 

---

## 🔒 Disclaimer

> **Note**:  
> This repository only contains **pseudocode representations** of the models described in the paper. **All data, training scripts, and real model implementations are proprietary and protected under Siemens Healthineers intellectual property.**  
> No actual code, image data, or trained model weights are included.

---

## 📦 Requirements

To run the pseudocode and adapt it to your environment, the following dependencies are recommended:

- `python >= 3.6.10`
- `pytorch >= 1.6.0`
- `jupyterlab` or `jupyter notebook`
- `torchio`
- `scikit-image`
- `scikit-learn`
- **GPU** (strongly recommended for training and inference)

> ⚠️ **Note**: The models *can* run on CPU, but execution is extremely slow and may cause system freezing due to the memory footprint and transformer complexity.

---

## 🧾 Dataset Preparation

To work with the pseudocode in this repository, structure your target (ground truth) and noisy (input) images in seperate folders with the same file name. The target and input images should be registered to avoid shifts in the model's outputs. Normalization between std 0.5 and mean 0.5 was performed for uniformity. 

##📁 Repository Layout

- `losses/`  
  Contains pseudocode implementations for various loss functions and evaluation metrics used during training and validation. This includes adversarial loss, L1 loss, Dice similarity metrics, and other custom components.

- `models/reGAN/`  
  Pseudocode for the latest version of the proposed model. This version includes architectural refinements for faster training and reduced computational cost while maintaining denoising quality.

- `models/regan_disk/`  
  Pseudocode for an earlier version of the model. This version emphasizes Transformer-based modules more heavily. While still accurate, it is slightly slower than the current optimized architecture.

> **Note:** Both model variants yield similar denoising performance. The primary difference lies in architectural efficiency—`reGAN` is more optimized and time-efficient than `regan_disk`.

- `train_reGAN/`  
  Contains pseudocode to illustrate the training pipeline, including loading datasets, initializing models, configuring loss functions, and running the training loop.


