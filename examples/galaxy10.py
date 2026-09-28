import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score
from tqdm import tqdm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def imnet_norm(t: torch.Tensor, device: torch.device):
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)
    return (t - mean) / std

from torchvision.models import ResNet50_Weights, ResNet18_Weights


class CSVPicDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        assert 'filepath' in self.df.columns and 'caption' in self.df.columns, \
            "CSV must include filepath and caption"
        self.filepaths = self.df['filepath'].tolist()
        self.labels = [int(x) for x in self.df['caption'].tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.filepaths[idx]
        img = Image.open(p).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = TF.to_tensor(img)
        return img_t, label, idx

# ------------------------------
# 2) SDSS -> DECals label mapping
# ------------------------------
sdss_to_decals_map = {1: 2, 2: 3, 3: 4, 4: 9, 5: 9, 6: 8}  

def load_sdss_h5_as_numpy(sdss_h5_path: str):
    """
    return (images_sdss_uint8, labels_sdss_mapped_int)
    """
    with h5py.File(sdss_h5_path, 'r') as f:
        raw_images = np.array(f['images'], dtype=np.float32)  # [N,H,W,3] 0-255
        raw_labels = np.array(f['ans'], dtype=np.int64)
    mapped = np.array([sdss_to_decals_map.get(int(l), -1) for l in raw_labels], dtype=np.int64)
    mask = mapped != -1
    images = raw_images[mask].clip(0, 255).astype(np.uint8)
    labels = mapped[mask]
    return images, labels

class NumpyImageDataset(Dataset):
    def __init__(self, images_uint8: np.ndarray, labels_int: np.ndarray, transform):
        self.images = images_uint8
        self.labels = labels_int
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        img = Image.fromarray(img_np)
        label = int(self.labels[idx])
        img_t = self.transform(img)
        return img_t, label, idx

# ------------------------------
# 3) Frequency-domain augmentation (Gradient-guided low-frequency amplitude mixing
# ------------------------------
@torch.no_grad()
def _predict_label_from_img_tensor(x01: torch.Tensor, model: nn.Module):
    """
    x01: [1,3,H,W] in [0,1], no normalization
    """
    device = next(model.parameters()).device
    logits = model(imnet_norm(x01, device))
    return int(logits.argmax(1).item())

def colorful_spectrum_mix(
    img1: Image.Image,
    img2: Image.Image,
    model: nn.Module = None,
    label: int = None,
    ratio: float = 1.0,
    per_channel: bool = False,
):
    """
    Low-frequency rectangular-window amplitude mixing:
    A_mix = (1 - d) * A1 + d * A2, while keeping the phase of img1.
    - img1, img2: PIL.Image (uint8)
    - model: classification model used to guide the gradients; if None,
      this degrades to a random-λ mixing.
    - label: supervision label; if None, we first predict the label of Y1.
    - ratio: area ratio of the low-frequency window (side length ~ sqrt(ratio)).
    - per_channel: whether d is normalized per channel (default False;
      using a single-channel mean is more stable).
    Returns: PIL.Image
    """
    # to [0,1]
    x1 = TF.to_tensor(img1).unsqueeze(0)  # [1,3,H,W]
    x2 = TF.to_tensor(img2).unsqueeze(0)
    _, C, H, W = x1.shape

    if model is None:
        lam = torch.rand(1).item() * 0.7
        X1 = torch.fft.fft2(x1, dim=(-2, -1))
        A1, P1 = torch.abs(X1), torch.angle(X1)
        X2 = torch.fft.fft2(x2, dim=(-2, -1))
        A2 = torch.abs(X2)
        h0, w0 = H // 2, W // 2
        h_crop = int(H * math.sqrt(ratio));  w_crop = int(W * math.sqrt(ratio))
        h1, h2 = h0 - h_crop // 2, h0 + h_crop // 2
        w1, w2 = w0 - w_crop // 2, w0 + w_crop // 2
        A1m = A1.clone()
        A1m[:, :, h1:h2, w1:w2] = lam * A2[:, :, h1:h2, w1:w2] + (1 - lam) * A1[:, :, h1:h2, w1:w2]
        Y = torch.fft.ifft2(A1m * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)
        out = (Y * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(out)

    device = next(model.parameters()).device
    x1 = x1.to(device); x2 = x2.to(device)

    # FFT
    X1 = torch.fft.fft2(x1, dim=(-2, -1))
    A1, P1 = torch.abs(X1), torch.angle(X1)
    X2 = torch.fft.fft2(x2, dim=(-2, -1))
    A2 = torch.abs(X2)

    # omega
    h0, w0 = H // 2, W // 2
    h_crop = int(H * math.sqrt(ratio));  w_crop = int(W * math.sqrt(ratio))
    h1, h2 = h0 - h_crop // 2, h0 + h_crop // 2
    w1, w2 = w0 - w_crop // 2, w0 + w_crop // 2

    was_training = model.training
    model.eval()

    # grad computing
    A1_req = A1.detach().clone().requires_grad_(True)
    Y1 = torch.fft.ifft2(A1_req * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)
    if label is None:
        y = torch.tensor([_predict_label_from_img_tensor(Y1, model)], device=device)
    else:
        y = torch.tensor([int(label)], device=device)

    logits = model(imnet_norm(Y1, device))
    ce = F.cross_entropy(logits, y)
    g = torch.autograd.grad(ce, A1_req, retain_graph=False, create_graph=False)[0].abs()  # [1,3,H,W]

    model.train(was_training)


    gw = g[:, :, h1:h2, w1:w2]
    if per_channel:
        gf = gw.view(1, C, -1)
        gmax = gf.max(-1, keepdim=True).values.view(1, C, 1, 1)
        gmin = gf.min(-1, keepdim=True).values.view(1, C, 1, 1)
        d = (gw - gmin) / (gmax - gmin + 1e-6)
    else:
        gmean = gw.mean(1, keepdim=True)  # [1,1,hc,wc]
        gf = gmean.view(1, -1)
        gmax = gf.max(-1, keepdim=True).values.view(1, 1, 1, 1)
        gmin = gf.min(-1, keepdim=True).values.view(1, 1, 1, 1)
        d = (gmean - gmin) / (gmax - gmin + 1e-6)
        d = d.expand(-1, C, -1, -1)
        d_flat = d.reshape(-1)
        mean = d_flat.mean()
        std = d_flat.std() + 1e-6
        d_norm = (d - mean) / std
        d = torch.sigmoid(d_norm).clamp(0.4, 0.7)


    A_mix = A1.clone()
    A_mix[:, :, h1:h2, w1:w2] = (1.0 - d) * A1[:, :, h1:h2, w1:w2] + d * A2[:, :, h1:h2, w1:w2]

    # IFFT -> PIL
    Y = torch.fft.ifft2(A_mix * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)
    out = (Y * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(out)

def maybe_mix_in_train_loop(
    model: nn.Module,
    batch_imgs: torch.Tensor,     # [B,3,224,224]
    batch_labels: torch.Tensor,   # [B]
    batch_idxs: torch.Tensor,     # [B]
    csv_dataset: CSVPicDataset,   
    sdss_images_uint8: np.ndarray,
    train_transform: transforms.Compose,
    mix_prob: float = 0.5,
    ratio: float = 0.25,
):
    """
    Perform cross-domain frequency-domain mixing for part of the samples
    during the training loop:
    - Reload img1 (DECals) as a PIL image from csv_dataset.filepaths.
    - Randomly pick img2 (SDSS) from sdss_images_uint8 as a PIL image.
    - First blend in pixel space with Image.blend(0.3), then apply
      frequency-domain mixing, and blend again with blend(0.2).
    - Finally apply train_transform to the mixed image and write it back
      to the corresponding position in batch_imgs.
    """
    B = batch_imgs.size(0)
    device = batch_imgs.device
    if sdss_images_uint8 is None or len(sdss_images_uint8) == 0 or mix_prob <= 0.0:
        return batch_imgs

    mask = (torch.rand(B, device=device) < mix_prob)
    if not mask.any():
        return batch_imgs

    sel = mask.nonzero(as_tuple=False).view(-1).tolist()
    for b in sel:
        if np.random.uniform(0, 1) >= 0.5:
            continue

        gidx = int(batch_idxs[b].item())
        img1 = Image.open(csv_dataset.filepaths[gidx]).convert('RGB').resize((224, 224))
        half_h = img1.size[0] // 2
        half_w = img1.size[1] // 2
        img1 = TF.center_crop(img1, (half_h, half_w))
        img1 = TF.resize(img1, (224, 224))

        k = np.random.randint(0, len(sdss_images_uint8))
        img2 = Image.fromarray(sdss_images_uint8[k]).convert('RGB').resize((224, 224))

        combined_img1 = Image.blend(img1, img2, 0.3)

        mixed_pil = colorful_spectrum_mix(
            img1, img2,
            model=model,
            label=int(batch_labels[b].item()),
            ratio=ratio,
            per_channel=False
        )

        mixed_pil = Image.blend(mixed_pil, combined_img1, 0.2)

        batch_imgs[b] = train_transform(mixed_pil).to(device)

    return batch_imgs


def load_model(model_name: str, num_classes: int):
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("model_name must be 'resnet50' or 'resnet18'")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model, loader, optimizer, criterion,
    csv_dataset: CSVPicDataset,
    sdss_images_uint8: np.ndarray,
    train_transform: transforms.Compose,
    mix_prob: float = 0.5,
    ratio: float = 0.25,
):
    model.train()
    running_loss = 0.0
    for imgs, labels, idxs in tqdm(loader, desc='Training', leave=False):
        imgs, labels, idxs = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True), idxs.cuda(non_blocking=True)

        imgs = maybe_mix_in_train_loop(
            model=model,
            batch_imgs=imgs,
            batch_labels=labels,
            batch_idxs=idxs,
            csv_dataset=csv_dataset,
            sdss_images_uint8=sdss_images_uint8,
            train_transform=train_transform,
            mix_prob=mix_prob,
            ratio=ratio
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels, _ in tqdm(loader, desc='Evaluating', leave=False):
        imgs = imgs.cuda(non_blocking=True)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend([int(l) for l in labels])

    acc = accuracy_score(y_true, y_pred)
    return acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_csv = "/data/galaxy10/train_demo.csv"
    test_csv  = "/data/galaxy10/test.csv"
    train_dataset = CSVPicDataset(train_csv, transform=train_transform)
    test_dataset  = CSVPicDataset(test_csv,  transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    sdss_h5 = "/data/Galaxy10.h5"
    sdss_images_uint8, sdss_labels_mapped = load_sdss_h5_as_numpy(sdss_h5)
    print(f"[SDSS] usable samples: {len(sdss_images_uint8)} "
          f"(after mapping {-1} filtered).")
    sdss_test_set = NumpyImageDataset(sdss_images_uint8, sdss_labels_mapped, test_transform)
    sdss_test_loader = DataLoader(
        sdss_test_set, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    num_classes = 10  
    model = load_model('resnet18', num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    for ep in range(epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            csv_dataset=train_dataset,
            sdss_images_uint8=sdss_images_uint8,
            train_transform=train_transform,
            mix_prob=0.3,  
            ratio=0.25       
        )
        print(f"Epoch [{ep+1}/{epochs}]  train_loss: {loss:.4f}")

        acc_csv = evaluate(model, test_loader)
        print(f"  -> CSV Test:  Acc={acc_csv:.4f}")

    acc_sdss = evaluate(model, sdss_test_loader)
    print(f"\n[Cross-domain: SDSS]  Acc={acc_sdss:.4f}")

if __name__ == "__main__":
    main()
