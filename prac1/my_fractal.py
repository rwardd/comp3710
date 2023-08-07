import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WIDTH = 1000
HEIGHT = 1000

x = 0
y = 0
ns = torch.zeros((WIDTH, HEIGHT))
ns = ns.to(device)
for i in range(0, 10000):
    r = torch.rand(1)
    if r < 0.1:
        x = 0.0 
        y = 0.16 * y
    elif r < 0.86:
        x = 0.85 * x + 0.04 * y
        y = -0.04 * x + 0.85 * y + 1.6
    elif r < 0.93:
        x = 0.2 * x - 0.26 * y 
        y = 0.23 * x + 0.22 * y + 1.6
    else:
        x = -0.15 * x + 0.22 * y
        y = 0.26 * x + 0.24 * y + 0.44
    iy, ix = int(WIDTH/ 2 + x * WIDTH/ 10), int(y * HEIGHT/ 12)
    ns[ix, iy] = 1


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

ns = ns.numpy()
plt.imshow(ns[::-1, :])
plt.tight_layout(pad=0)
plt.show()
