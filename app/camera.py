import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def capture_mask_from_camera(model_path, device='cpu'):
    camera = cv2.VideoCapture(0)
    desired_size = 256
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()  

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Pobranie wymiarów obrazu
        height, width = frame.shape[:2]
        x_start = (width - desired_size) // 2
        y_start = (height - desired_size) // 2
        x_end = x_start + desired_size
        y_end = y_start + desired_size

        # Przycinanie ramki do 256x256
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        
        # Preprocessing i przewidywanie maski
        mask = get_pred_from_frame(cropped_frame, model, device)

        # Nakładanie maski na obraz
        masked_image = get_mask_on_photo(cropped_frame, mask)

        # Wyświetlanie nałożonej maski
        frame[y_start:y_end, x_start:x_end] = masked_image
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
        
        # Wyświetlanie podglądu kamery z maską
        cv2.imshow("Podgląd z kamery", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

def get_pred_from_frame(frame, model, device='cpu'):
    # Wstępne przetwarzanie obrazu
    image = preprocess_img_from_frame(frame)
    image = image.unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(image)
    
    # Przetwarzanie wyjścia modelu
    output_image = output.squeeze(0).cpu().numpy()
    output_image = (output_image > 0.5).astype(np.uint8)
    output_image = np.transpose(output_image, (1, 2, 0))
    
    return output_image

def preprocess_img_from_frame(frame):
    # Zmiana rozmiaru i normalizacja obrazu (dostosowanie do potrzeb modelu)
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.as_tensor(frame, dtype=torch.float) / 255.0
    frame = frame.permute(2, 0, 1) # Konwersja na tensora PyTorch
    return frame

def get_mask_on_photo(img, pred):
    mask = np.zeros_like(img)
    mask[:, :, 2] = pred[:, :, 0] * 255  # bierzemy czerwony jako usta, tu wejście BGR bo to obraz z kamery 
    alpha = 0.2
    masked_image = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
    return masked_image

if __name__ == '__main__':
    capture_mask_from_camera(r'C:\Users\aleks\OneDrive\Documents\LIPS SEGMENTATION\resnet_models\model_3_11_2.pth')
