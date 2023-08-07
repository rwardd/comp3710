import torch
import numpy as np
import matplotlib.pyplot as plt

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('mps')

Y, X = np.mgrid[-0.3:0:0.0001, -1:-0.5:0.0001]

x = torch.tensor(X, dtype=torch.double)
y = torch.tensor(Y, dtype=torch.double)
z = torch.complex(x, y) #important!
zs = z #torch.zeros_like(z)
ns = torch.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

for i in range(200):
    #Compute the new values of z: z^2 + x
    zs_ = zs*zs + z # TODO: fails here
    #Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    #Update variables to compute
    ns += not_diverged.type(torch.FloatTensor)
    zs = zs_

#plot
fig = plt.figure(figsize=(16,10))

def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a


plt.imshow(processFractal(ns.numpy()))
plt.tight_layout(pad=0)
plt.show()
