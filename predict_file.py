import torch
from model_Unet import UNET
from dataset_file import *
from train_file import * 
import matplotlib.pyplot as plt

def preprocess_img(image_path, new_size = (256,256)):
    image = cv2.imread(image_path)
    img = cv2.cvtColor(cv2.resize(image, new_size), cv2.COLOR_BGR2RGB)
        
    img = img.astype(float)
    img = torch.as_tensor(img, dtype=torch.float) / 255.0
    img = img.permute(2, 0, 1)  # Permutujemy do formatu (C, H, W)
    return img

def get_pred(img_path, model_path):

    model = UNET()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)

    image = preprocess_img(img_path)
    image = image.unsqueeze(0).float().to(DEVICE)
        
    model.eval()
    with torch.no_grad():  # Wyłączamy gradienty
        output = torch.sigmoid(model(image))

    output_image = output.squeeze(0).cpu().numpy()  # Usuń wymiar batcha i przenieś na CPU
    output_image = (output_image > 0.5).astype(np.uint8)  # Ustal próg dla maski
    output_image = np.transpose(output_image, (1, 2, 0))  # Zamień na szerokość, wysokość, kanały
    return output_image

def get_mask_on_photo(img, pred):
    mask = np.zeros_like(img)
    mask[:, :, 0] = pred[:, :, 0] * 255  # bierzemy czerwony jako usta
    alpha = 0.2
    masked_image = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
    return masked_image

def print_prediction(path_img, model_path):
    img = cv2.imread(path_img)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = get_pred(path_img, model_path)
    masked_image = get_mask_on_photo(img, mask)

    fig = plt.figure(figsize = (16,5))
    a = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(img)
    a.set_title('Input')
    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(mask)
    a.set_title('Output')
    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(masked_image)
