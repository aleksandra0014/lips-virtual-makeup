import os 
import cv2 
import numpy as np 

label_list = ['u_lip', 'l_lip']

folder_base = 'masks'
folder_save = 'proper_masks'

img_num = 2000

os.chdir(os.path.join(os.getcwd())) 

print(os.listdir())

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

make_folder(folder_save)

for k in range(img_num):
    im_base = np.zeros((512,512))
    is_empty = True
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(k).rjust(5, '0') + '_' + label + '.png')
        if os.path.exists(filename.strip(' ')):
            is_empty = False
            im = cv2.imread(filename)
            im = im[:, :, 0] # tylko pierwszy kanał
            im_base[im != 0 ] = 255  # tam gdzie piksel nie jest zerem ustawiamy biały kolor - czyli tam gdzie jest maska 
        
        if not is_empty:
            filename_save = os.path.join(folder_save, str(k) + '.png')
            print(filename_save)
            cv2.imwrite(filename_save, im_base)
