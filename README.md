# Virtual Lipstic Try-On 
The Virtual Lipstick Try-On project focused on image processing and deep neural networks. I developed a custom U-Net model for lip segmentation using PyTorch, enabling precise identification of lips in images. Following this, I built an application in CustomTkinter to allow users to virtually apply different lipstick shades. 

## Features

- **Facial Analysis**: Automatically detects lips on an image or video feed.
- **Customization**: Enables users to choose different lipstick shades, gloss, or matte finishes.
- **Real-Time Visualization**: Instantly preview makeup effects live.
- **Result Saving**: Option to save the applied makeup as an image for sharing or future reference.

## Technologies Used
- **Programming Language**: Python
- **Frameworks and Libraries**:
  - OpenCV (for image processing)
  - Pytorch (for segmentation model)
  - customtkinter (for GUI application)
  - NumPy/Pandas (for data analysis)
 
## Demo 
If you want to see how project works: https://drive.google.com/file/d/16yZjVTPLCx5sXE4H8cpWkZwUf_zX-AxJ/view?usp=sharing

 ```bash
  - lips-virtual-makeup/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── README.md             # Project description
├── notebooks/            # Model testing and analysing
├── own_model/            # Creating an own UNet model
├── dataset_preprocess/   # Files for preprocess images
├── doc/                  # Presentation and report about project

