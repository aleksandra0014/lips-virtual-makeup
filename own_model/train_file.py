import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model_Unet import UNET
import torch.nn.functional as F  
import matplotlib.pyplot as plt

def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16


def train_fn(train_loader, val_loader, model, optimizer, num_epochs, model_path):

    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Najlepszy wynik walidacyjny, początkowo ustawiony na nieskończoność

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc='Training loop'):
            images = images.to(device=DEVICE)
            masks = masks.float().to(device=DEVICE)
            
            optimizer.zero_grad()
            predictions = model(images)
            #loss = loss_fn(predictions, masks)
            bce_loss = F.binary_cross_entropy_with_logits(predictions, masks)
            dice = dice_loss(predictions, masks)
            loss = bce_loss + dice
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * masks.size(0)  #loss.item daje średnia starte wiec mnoze przez liczbe probek w batchu
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation loop'):
                images = images.to(device=DEVICE)
                masks = masks.float().to(device=DEVICE)
                predictions = model(images)
                #loss = loss_fn(predictions, masks)
                bce_loss = F.binary_cross_entropy_with_logits(predictions, masks)
                dice = dice_loss(predictions, masks)
                loss = bce_loss + dice
                running_loss += loss.item() * masks.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)  # Zapis najlepszych wag
            print(f"Best model weights saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    # wykresiki
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

if __name__ == '__main__':
    pass
