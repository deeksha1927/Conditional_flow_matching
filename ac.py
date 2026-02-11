# train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

# --------- Config ---------
savedir = "models/ear"
os.makedirs(savedir, exist_ok=True)

batch_size = 128            # per-GPU batch size
n_epochs = 100
lr = 1e-4
num_workers = 4

# --------- DDP helpers ---------
def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.barrier()
    dist.destroy_process_group()

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

# --------- Main ---------
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cudnn.benchmark = True

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Dataset + DistributedSampler
    trainset = datasets.ImageFolder(
        root="/store01/flynn/darun/uerc2023/data/public/",
        transform=transform
    )
    sampler = DistributedSampler(trainset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,          # per-GPU
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Model / Optimizer / CFM
    sigma = 0.0
    model = UNetModel(
        dim=(3, 128, 128),
        num_channels=128,
        num_res_blocks=1,
        num_classes=1310,
        class_cond=True,
        use_new_attention_order=True,
        use_scale_shift_norm=True
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    FM = ConditionalFlowMatcher(sigma=sigma)

    for epoch in range(n_epochs):
        model.train()
        sampler.set_epoch(epoch)  # shuffles shards differently each epoch

        for i, (x1, y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            x1 = x1.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)  # same device as inputs

            vt = model.module(t, xt, y)  # call the wrapped module explicitly
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()

            if is_main_process() and (i % 50 == 0):
                print(f"epoch: {epoch+1}/{n_epochs}, step: {i}, loss: {loss.item():.4f}")

        # checkpoint every 10 epochs (main process only)
        if is_main_process() and ((epoch + 1) % 10 == 0):
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            ckpt_path = os.path.join(savedir, f"model_epoch_{epoch+1}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"[checkpoint] saved: {ckpt_path}")

    cleanup_ddp()

if __name__ == "__main__":
    main()

