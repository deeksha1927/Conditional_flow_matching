import os
import torch
import torchdiffeq
from torchvision.utils import save_image
from torchcfm.models.unet.unet import UNetModelWrapper

# -----------------------
# CONFIG
# -----------------------
CHECKPOINT = "/users/darun/afs/conditional-flow-matching/models/ear/model_epoch_70.pt"
BASE_OUTDIR = "/store01/flynn/darun/cfm_generated"   # outputs: BASE_OUTDIR/0001 ... /1310
NUM_CLASSES = 1310

SEED_START = 0
SEED_END   = 50   # inclusive

IMG_SIZE = 128
DEVICE = "cuda"   # uses cpu if cuda unavailable

# Must match training architecture
NUM_CHANNELS = 128
NUM_RES_BLOCKS = 1

# Safer than dopri5 for long runs
SOLVER = "dopri5"   # options: "euler", "rk4", "dopri5"
# -----------------------


@torch.no_grad()
def sample_one(model, device, label: int, seed: int):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    y = torch.tensor([label], device=device, dtype=torch.long)
    x0 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    def f(t, x):
        return model(t, x, y)

    tspan = torch.tensor([0.0, 1.0], device=device)

    if SOLVER == "euler":
        traj = torchdiffeq.odeint(f, x0, tspan, method="euler")
    elif SOLVER == "rk4":
        traj = torchdiffeq.odeint(f, x0, tspan, method="rk4")
    elif SOLVER == "dopri5":
        traj = torchdiffeq.odeint(f, x0, tspan, method="dopri5", atol=1e-4, rtol=1e-4)
    else:
        raise ValueError(f"Unknown solver: {SOLVER}")

    xT = traj[-1][0]
    img = (xT * 0.5 + 0.5).clamp(0, 1)  # unnormalize to [0,1]
    return img


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Recreate the exact model used in training
    model = UNetModelWrapper(
        dim=(3, IMG_SIZE, IMG_SIZE),
        num_channels=NUM_CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS,
        num_classes=NUM_CLASSES,
        class_cond=True,
        use_new_attention_order=True,
        use_scale_shift_norm=True,
        use_checkpoint=False
    ).to(device)

    # Load checkpoint
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    os.makedirs(BASE_OUTDIR, exist_ok=True)

    # Folder classes are 1-based (0001..1310), but labels are 0-based (0..1309)
    for cls in range(1, NUM_CLASSES + 1):
        outdir = os.path.join(BASE_OUTDIR, f"{cls:04d}")
        os.makedirs(outdir, exist_ok=True)

        label = cls - 1  # <-- key mapping

        for seed in range(SEED_START, SEED_END + 1):
            img = sample_one(model, device, label=label, seed=seed)
            savepath = os.path.join(outdir, f"seed{seed:04d}.png")
            save_image(img, savepath)

            # keep long runs stable
            del img
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"Done folder {cls:04d} (label {label}) -> {outdir}")


if __name__ == "__main__":
    main()

