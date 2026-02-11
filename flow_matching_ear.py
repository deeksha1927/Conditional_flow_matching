import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
from torch.utils.data import DataLoader

savedir = "models/ear"
os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 128
n_epochs = 100


# Define transformations (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),          # Resize to 128x128
    transforms.ToTensor(),                  # Convert images to tensor with shape [3, 128, 128]
    transforms.Normalize((0.5, 0.5, 0.5),   # Normalize each RGB channel
                         (0.5, 0.5, 0.5))
])

# Load dataset using ImageFolder
trainset = datasets.ImageFolder(
    root="/store01/flynn/darun/uerc2023/data/public/",  # Path to your main data folder (my_data)
    transform=transform      # Apply transformations
)


train_loader = DataLoader(trainset, batch_size=32, shuffle=True)


#################################
#    Class Conditional CFM
#################################

sigma = 0.0
model = UNetModel(
    dim=(3, 128, 128), num_channels=128, num_res_blocks=1, num_classes=1310, class_cond=True, use_new_attention_order=True, use_scale_shift_norm=True
).to(device)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)
# Users can try target FM by changing the above line by
# FM = TargetConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)



for epoch in range(n_epochs):
    print(epoch)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data[0].to(device)
        y = data[1].to(device)
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt, y)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
        
        # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_path = f"{savedir}/model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved at epoch {epoch+1} -> {save_path}")
