## DDPM consists basic UNet, GaussianDiffusion, and a few other components.
## Test on stanford cars dataset.

import torch
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt

def show_images(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15,15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols)+1, cols, i+1)
        plt.imshow(img[0])

if __name__ == '__main__':
    train_dataset = torchvision.datasets.StanfordCars(root='.', split='train', download=True)
    show_images(train_dataset)