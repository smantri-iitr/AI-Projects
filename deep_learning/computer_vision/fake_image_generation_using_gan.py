import os
import math
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils as vutils

# -----------------------------
# Config
# -----------------------------
SEED           = 42
DATA_ROOT      = "./data"        # where CelebA will be stored
SAMPLES_DIR    = "./samples"
CKPT_DIR       = "./checkpoints"

IMAGE_SIZE     = 64              # DCGAN paper uses 64
NC             = 3               # number of image channels
NZ             = 128             # size of latent z vector
NGF            = 64              # feature maps in generator
NDF            = 64              # feature maps in discriminator

BATCH_SIZE     = 128
NUM_EPOCHS     = 25
LR             = 2e-4
BETA1          = 0.5
BETA2          = 0.999
LABEL_SMOOTH   = 0.9             # real label smoothing
SAVE_EVERY     = 500             # iterations between sample image saves

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.CenterCrop(178),      # CelebA is 178x218
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# target_type='attr' just to satisfy the API; we ignore the labels
dataset = datasets.CelebA(
    root=DATA_ROOT, split="train", target_type="attr",
    transform=transform, download=True
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)

# -----------------------------
# Models (DCGAN)
# -----------------------------
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)

class Generator(nn.Module):
    def __init__(self, nz=NZ, ngf=NGF, nc=NC):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: [B, nz, 1, 1]
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # output [-1, 1]
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=NDF, nc=NC):
        super().__init__()
        self.main = nn.Sequential(
            # input: [B, nc, 64, 64]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)  # logits
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1)  # [B]

netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
netG.apply(weights_init_dcgan)
netD.apply(weights_init_dcgan)

# Loss & optimizers
criterion = nn.BCEWithLogitsLoss()

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))

fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)

# -----------------------------
# Training
# -----------------------------
global_step = 0
print(f"Starting training on {DEVICE} with {len(dataset)} images")

for epoch in range(1, NUM_EPOCHS+1):
    for i, (imgs, _) in enumerate(dataloader):
        netD.train()
        netG.train()
        real = imgs.to(DEVICE)

        b_size = real.size(0)
        real_labels = torch.full((b_size,), LABEL_SMOOTH, device=DEVICE)
        fake_labels = torch.zeros(b_size, device=DEVICE)

        # -------------------------
        # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
        # -------------------------
        optimizerD.zero_grad(set_to_none=True)

        logits_real = netD(real)
        lossD_real = criterion(logits_real, real_labels)

        noise = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
        fake = netG(noise).detach()  # detach so G isn't updated here
        logits_fake = netD(fake)
        lossD_fake = criterion(logits_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # -------------------------
        # (2) Update G: maximize log(D(G(z)))  <=> minimize BCE(D(G(z)), 1)
        # -------------------------
        optimizerG.zero_grad(set_to_none=True)

        noise2 = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
        fake2 = netG(noise2)
        logits_fake2 = netD(fake2)
        lossG = criterion(logits_fake2, real_labels)  # "real" labels to fool D
        lossG.backward()
        optimizerG.step()

        # -------------------------
        # Logging / sampling
        # -------------------------
        if global_step % 100 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Epoch {epoch}/{NUM_EPOCHS} | Step {i}/{len(dataloader)} | "
                  f"D: {lossD.item():.4f} (R {lossD_real.item():.4f} / F {lossD_fake.item():.4f}) | "
                  f"G: {lossG.item():.4f}")

        if global_step % SAVE_EVERY == 0:
            with torch.no_grad():
                netG.eval()
                fake_fixed = netG(fixed_noise).detach().cpu()
            grid_path = os.path.join(SAMPLES_DIR, f"epoch{epoch:02d}_step{global_step:06d}.png")
            vutils.save_image(fake_fixed, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
            print(f"Saved samples to {grid_path}")

        global_step += 1

    # Save checkpoints each epoch
    ckpt_path = os.path.join(CKPT_DIR, f"dcgan_celeba_epoch{epoch:02d}.pt")
    torch.save({
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optG": optimizerG.state_dict(),
        "optD": optimizerD.state_dict(),
        "config": {
            "IMAGE_SIZE": IMAGE_SIZE, "NZ": NZ, "NGF": NGF, "NDF": NDF
        }
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

print("Training complete!")

# -----------------------------
# Inference helper (optional)
# -----------------------------
# After training, you can run this block to sample images:
# python dcgan_celeba.py --sample path_to_checkpoint.pt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="")
    parser.add_argument("--n", type=int, default=64, help="number of images to sample")
    args = parser.parse_args()

    if args.sample:
        ckpt = torch.load(args.sample, map_location=DEVICE)
        netG.load_state_dict(ckpt["netG"])
        netG.eval()
        with torch.no_grad():
            z = torch.randn(args.n, NZ, 1, 1, device=DEVICE)
            fakes = netG(z).cpu()
        # save a grid
        out_path = os.path.join(SAMPLES_DIR, f"sample_grid_{os.path.basename(args.sample).replace('.pt','')}.png")
        vutils.save_image(fakes, out_path, nrow=int(math.sqrt(args.n)), normalize=True, value_range=(-1,1))
        print(f"Saved samples to {out_path}")
