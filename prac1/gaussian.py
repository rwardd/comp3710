import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(mps_device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X)
y = torch.Tensor(Y)

x = x.to(device)
y = y.to(device)

z = torch.sin(3*x + y) * torch.exp(-(x**2+y**2)/2.0)

plt.imshow(z.numpy())
plt.tight_layout()
plt.show()



