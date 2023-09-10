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

net = GAN()
net = net.to(device)
lr = 0.0002
epochs = 5
criterion = nn.BCELoss()
latent_fixed = torch.rand(64, latent_size, 1, 1, device=device)
real_label = 1
fake_label = 0
sample_dir="pytorch_generated"
os.makedirs(sample_dir, exist_ok=True)
optimizer_d = torch.optim.Adam(net.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_g = torch.optim.Adam(net.generator.parameters(), lr=lr, betas=(0.5, 0.999))

img_list = []
g_losses = []
d_losses = []
num_iterations = 0

def save_samples(index, fake_images):
    fake_fname = 'images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)


print("Starting Training")
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        net.discriminator.zero_grad()
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = net.discriminator(real_data).view(-1)
        error_d_real = criterion(output, label)
        error_d_real.backward()
        d_x = output.mean().item()

        noise = torch.randn(b_size, latent_size, 1, 1, device=device)
        fake_data = net.generator(noise)
        label.fill_(fake_label)
        output = net.discriminator(fake_data.detach()).view(-1)
        error_d_fake = criterion(output, label)
        error_d_fake.backward()
        d_g_z1 = output.mean().item()
        error_d = error_d_real + error_d_fake
        optimizer_d.step()

        net.generator.zero_grad()
        label.fill_(real_label)
        output = net.discriminator(fake_data).view(-1)
        error_g = criterion(output, label)
        error_g.backward()
        d_g_z2 = output.mean().item()
        optimizer_g.step()
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(train_loader),
                     error_d.item(), error_g.item(), d_x, d_g_z1, d_g_z2))
        g_losses.append(error_g.item())
        d_losses.append(error_d.item())
        if (num_iterations % 500 == 0) or ((epoch == epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = net.generator(latent_fixed).detach().cpu()
            save_samples(num_iterations, fake)
        num_iterations += 1

plt.figure(figsize=(10,5))
plt.title("Gen and Discrim loss during training")
plt.plot(g_losses, label='g')
plt.plot(d_losses, label='d')
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

torch.save(net.generator.state_dict(), 'Generator_run1.pth')
torch.save(net.discriminator.state_dict(), 'Discriminator_run1.pth')


