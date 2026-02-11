
import os
import torch
import torchdiffeq
from torchvision.utils import save_image

from torchcfm.models.unet import UNetModel


# --------------------------------------------------
# Load model
# --------------------------------------------------
checkpoint = "/users/darun/afs/conditional-flow-matching/models/ear/model_epoch_80.pt"
save_dir = "/users/darun/afs/conditional-flow-matching/generated_images/"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetModel(
    dim=(3, 128, 128),
    num_channels=128,
    num_res_blocks=1,
    num_classes=1310,
    class_cond=True,
    use_new_attention_order=True,
    use_scale_shift_norm=True
).to(device)

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()


# --------------------------------------------------
# Disable checkpointing (critical fix)
# --------------------------------------------------
for m in model.modules():
    if hasattr(m, "use_checkpoint"):
        m.use_checkpoint = False



# --------------------------------------------------
# Vector field consistent with training
# --------------------------------------------------
def vector_field(t, x, y):
    return model.forward(t, x, y)



# --------------------------------------------------
# Sampling function
# --------------------------------------------------
def sample_cfm(y_label: int, num_samples: int = 8):

    y = torch.full((num_samples,), y_label, dtype=torch.long, device=device)
    x0 = torch.randn(num_samples, 3, 128, 128, device=device)

    tspan = torch.linspace(0., 1., 2, device=device)

    with torch.no_grad():
        traj = torchdiffeq.odeint(
            lambda t, x: vector_field(t, x, y),
            x0,
            tspan,
            method="dopri5",
            rtol=1e-4,
            atol=1e-4,
        )

    xT = traj[-1]     # final samples

    xT = (xT.clamp(-1, 1) + 1) * 0.5  # convert to [0,1]

    paths = []
    for i in range(num_samples):
        out_path = os.path.join(save_dir, f"class_{y_label}_{i}.png")
        save_image(xT[i], out_path)
        paths.append(out_path)

    return xT, paths



# --------------------------------------------------
# Example
# --------------------------------------------------
if __name__ == "__main__":
    imgs, paths = sample_cfm(10, num_samples=8)
    print("\nSaved images:")
    for p in paths:
        print(" →", p)
