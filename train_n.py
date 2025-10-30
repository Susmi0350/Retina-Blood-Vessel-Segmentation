import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Paths — fix for Windows """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_image_path = os.path.join(BASE_DIR, "new_data", "train", "image")
    train_mask_path = os.path.join(BASE_DIR, "new_data", "train", "mask")
    valid_image_path = os.path.join(BASE_DIR, "new_data", "test", "image")
    valid_mask_path = os.path.join(BASE_DIR, "new_data", "test", "mask")

    print("Base directory:", BASE_DIR)
    print("Train image path:", train_image_path)
    print("Train mask path:", train_mask_path)
    print("Valid image path:", valid_image_path)
    print("Valid mask path:", valid_mask_path)

    """ Check folder contents """
    print("Train image folder contents:", os.listdir(train_image_path))
    print("Train mask folder contents:", os.listdir(train_mask_path))
    print("Valid image folder contents:", os.listdir(valid_image_path))
    print("Valid mask folder contents:", os.listdir(valid_mask_path))

    """ Load dataset """
    train_x = sorted(glob(os.path.join(train_image_path, "**", "*.png"), recursive=True))
    train_y = sorted(glob(os.path.join(train_mask_path, "**", "*.png"), recursive=True))
    valid_x = sorted(glob(os.path.join(valid_image_path, "**", "*.png"), recursive=True))
    valid_y = sorted(glob(os.path.join(valid_mask_path, "**", "*.png"), recursive=True))

    print(f"Train images: {len(train_x)}, Train masks: {len(train_y)}")
    print(f"Valid images: {len(valid_x)}, Valid masks: {len(valid_y)}")

    if len(train_x) == 0 or len(train_y) == 0 or len(valid_x) == 0 or len(valid_y) == 0:
        raise ValueError("No dataset files found! Check folder paths and filenames.")

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model & Training Setup """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint: {checkpoint_path}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
