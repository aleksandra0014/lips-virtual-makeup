import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LipsSet(Dataset):
    def __init__(self, parent_dir, image_dir, mask_dir):

        self.mask_list = glob.glob(parent_dir+'/'+mask_dir+'/*')
        self.mask_list.sort()
        self.image_list = []

        # an image exists for every mask
        for path in self.mask_list:
            self.image_list.append(path.replace(mask_dir, image_dir))
        self.mask_list = self.mask_list
                
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.resize(mask, (256, 256)), cv2.COLOR_RGBA2GRAY)
        mask = mask.reshape((256, 256, 1))  # Zmiana kształtu na (256, 256, 1)

        img = img.astype(float)
        img = torch.as_tensor(img, dtype=torch.float) / 255.0
        img = img.permute(2, 0, 1)  # Permutujemy do formatu (C, H, W)

        cls_mask_1 = np.where(mask[..., 0] > 50, 1, 0).astype('float')

        mask = torch.as_tensor([cls_mask_1], dtype=torch.float)  

        return img, mask

def train_val_test_split(dataset):
    train_dataset = torch.utils.data.Subset(dataset, range(0, int(0.8 * len(dataset))))
    val_dataset = torch.utils.data.Subset(dataset, range(int(0.8*len(dataset)), int(0.9*len(dataset))))
    test_dataset = torch.utils.data.Subset(dataset, range(int(0.9*len(dataset)), len(dataset)))
    return train_dataset, val_dataset, test_dataset
