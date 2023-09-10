#CelebA GAN
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = "./img_align_celeba"

image_size = 64
batch_size = 128
latent_size = 128

# Training sets
train_transformer = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = ImageFolder(root=data_dir, transform =train_transformer)
train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=3, pin_memory=True)

def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator = self._make_discriminator()
        self.generator = self._make_generator()

    def _make_generator_block(self, in_planes, planes, stride=2, padding=1):
        layers = []
        layers.append(nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(True))
        return layers
    
    def _make_generator(self):
        blocks = []
        blocks.extend(self._make_generator_block(latent_size, 512, 1, 0))
        blocks.extend(self._make_generator_block(512, 256, 2, 1))
        blocks.extend(self._make_generator_block(256, 128, 2, 1))
        blocks.extend(self._make_generator_block(128, 64, 2, 1))
        blocks.append(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False))
        blocks.append(nn.Tanh())
        return nn.Sequential(*blocks)


    def _make_discriminator_block(self, in_planes, planes, stride=2):
        layers = []
        layers.append(nn.Conv2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def _make_discriminator(self):
        blocks = []
        blocks.extend((self._make_discriminator_block(3, 64, 2)))
        blocks.extend((self._make_discriminator_block(64, 128, 2)))
        blocks.extend((self._make_discriminator_block(128, 256, 2)))
        blocks.extend((self._make_discriminator_block(256, 516, 2)))
        blocks.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False))
        blocks.append(nn.Flatten())
        blocks.append(nn.Sigmoid())
        return nn.Sequential(*blocks)

net = GAN()
net.to(device)
print(net.discriminator)
print(net.generator)

xb = torch.randn(batch_size, latent_size, 1, 1)
fake_images = net.generator(xb.cuda())
print(fake_images.shape)
show_images(fake_images.cpu())
