import os
import torch
import torchdiffeq

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper



# ---------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------
savedir = "/users/darun/afs/conditional-flow-matching/models/ear"
sampledir = "/users/darun/afs/conditional-flow-matching/generated_images/"
os.makedirs(savedir, exist_ok=True)
os.makedirs(sampledir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = datasets.ImageFolder(
    root="/store01/flynn/darun/uerc2023/data/public/",
    transform=transform
)

num_classes = len(trainset.classes)
print("Number of classes:", num_classes)

train_loader = DataLoader(trainset, batch_size=32, shuffle=True)


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
sigma = 0.0

model = UNetModelWrapper(
    dim=(3,128,128),
    num_channels=128,
    num_res_blocks=1,
    num_classes=num_classes,
    class_cond=True,
    use_new_attention_order=True,
    use_scale_shift_norm=True,
    use_checkpoint=False
).to(device)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
FM = ConditionalFlowMatcher(sigma=sigma)

n_epochs = 100

for epoch in range(n_epochs):
    model.train()
    
    for i, (x1, y) in enumerate(train_loader):

        x1 = x1.to(device)
        y  = y.to(device)

        x0 = torch.randn_like(x1)

        # Conditional Flow Matching target
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        # Model pred flow
        vt = model(t, xt, y)

        # Loss
        loss = ((vt - ut)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch+1}/{n_epochs}, step {i}, loss={loss.item():.4f}", end="\r")

    print()  # newline after epoch

    # -----------------------------------------------------------------
    # GENERATE SAMPLES AFTER EVERY EPOCH
    # -----------------------------------------------------------------
    epoch_dir = os.path.join(sampledir, f"epoch_{epoch+1:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for sample_id in range(3):          # <-- generate 3 samples
            savepath = os.path.join(epoch_dir, f"class1_sample{sample_id}.png")

            y = torch.tensor([1], device=device, dtype=torch.long)  # <-- class 1 only
            x0 = torch.randn(1, 3, 128, 128, device=device)

            traj = torchdiffeq.odeint(
                lambda t, x: model.forward(t, x, y),
                x0,
                torch.linspace(0,1,2,device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5"
            )

            xT = traj[-1][0]
            img = (xT * 0.5 + 0.5).clamp(0,1)

            save_image(img, savepath)

    print(f"Saved 3 samples of class 1 for epoch {epoch+1} → {epoch_dir}")

    # -----------------------------------------------------------------
    # Save model every 10 epochs
    # -----------------------------------------------------------------
    if (epoch + 1) % 10 == 0:
        path = f"{savedir}/model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), path)
        print(f"Saved model → {path}")
