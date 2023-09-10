#CelebA GAN
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        blocks.extend((self._make_discriminator_block(256, 512, 2)))
        blocks.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False))
        blocks.append(nn.Flatten())
        blocks.append(nn.Sigmoid())
        return nn.Sequential(*blocks)

def train_gan_discriminator(net, images, optimizer):
    optimizer.zero_grad()

    real_preds = net.discriminator(images)
    real_targets = torch.ones(images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = net.generator(latent)

    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = net.discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    optimizer.step()
    return loss.item(), real_score, fake_score

def train_gan_generator(net, optimizer):
    optimizer.zero_grad()
    latent = torch.randn(batch_size, latent_size, 1,1, device=device)
    fake_images = net.generator(latent)

    # Try to fool the discriminator
    preds = net.discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator 
    loss.backward()
    optimizer.step()
    return loss.item()

sample_dir = 'generated_run2'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = net.generator(latent_tensors)
    fake_fname = 'images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)

    if show:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)


def fit(net, epochs, lr, start_index):
    torch.cuda.empty_cache()

    losses_generator = []
    losses_discriminator = []
    real_scores = []
    fake_scores = []
    loss_d = 0
    loss_g = 0
    real_score = 0
    fake_score = 0


    optimizer_descriminator = torch.optim.Adam(net.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(net.generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            real_images = data[0]
            real_images = real_images.cuda()
            # Train discriminator
            loss_d, real_score, fake_score = train_gan_discriminator(net, real_images, optimizer_descriminator)
            # Train generator
            loss_g = train_gan_generator(net, optimizer_generator)

        # Record losses & scores
        losses_generator.append(losses_generator)
        losses_discriminator.append(losses_discriminator)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_index, fixed_latent, show=False)


    return losses_generator, losses_discriminator, real_scores, fake_scores

net = GAN()
net = net.to(device)
lr = 0.00025
epochs = 60
history = fit(net, epochs, lr, 1)
losses_generator, losses_discriminator, real_scores, fake_scores = history
plt.plot(losses_discriminator, '-')
plt.plot(losses_generator, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discrim', 'Gen'])
plt.show()
torch.save(net.generator.state_dict(), 'Generator1.pth')
torch.save(net.discriminator.state_dict(), 'Discriminator1.pth')


