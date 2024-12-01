import customtkinter as ctk
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from model_Unet import UNET
import os
from datetime import datetime

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class WelcomeScreen(ctk.CTk):
    def __init__(self, on_continue_callback):
        super().__init__()
        self.title("Welcome to Onllipstick!")
        self.geometry("600x500")
        self.center_window(650, 500)
        
        # Logo
        logo_image = ctk.CTkImage(
            dark_image=Image.open("logo2.png"),
            size=(300, 150)
        )
        logo_label = ctk.CTkLabel(self, text="", image=logo_image)
        logo_label.pack(pady=20)
        
        # Welcome text
        welcome_label = ctk.CTkLabel(
            self,
            text=f"Choose perfect lipstick for you. \n Try on diffrent colors and check how it suits you.",
            font=("Helvetica", 20),
            text_color="white",
            justify="center"
        )
        welcome_label.pack(pady=20)
        
        # Buttons
        continue_button = ctk.CTkButton(
            self,
            text="Continue",
            width=200,
            height=50,
            corner_radius=10,
            fg_color=rgb_to_hex((181, 49, 93)),
            command=on_continue_callback
        )
        continue_button.pack(pady=10)
        
        quit_button = ctk.CTkButton(
            self,
            text="Exit",
            width=200,
            height=50,
            corner_radius=10,
            fg_color="red",
            command=self.close
        )
        quit_button.pack(pady=10)

    def close(self):
        self.destroy()

    def center_window(self, width, height):
        """
        Wyśrodkowanie okna na ekranie.
        """
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

class MaskApp(ctk.CTk):
    def __init__(self, model_path, color=[39, 2, 255], device='cpu'):
        super().__init__()
        self.title("Mask Application with CustomTkinter")
        self.geometry("700x550")
        self.color = color
        self.device = device
        self.desired_size = 256
        self.alpha = 0.3  
        self.video_paused = False  

        self.model = torch.load(model_path, map_location=device)
        self.model = self.model.to(device)
        self.model.eval()

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None  

        
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=5, column=0, padx=40, pady=30)

        
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=7, column=0, pady=10)  

        self.save_button = ctk.CTkButton(
            button_frame,
            text="Save photo!",
            fg_color='black',
            width=120,
            height=50,
            corner_radius=10,
            command=self.save_photo
        )
        self.save_button.grid(row=0, column=0, padx=5, pady=5)  

        self.pause_button = ctk.CTkButton(
            button_frame,
            text="Pause/Resume Video",
            fg_color='black',
            width=120,
            height=50,
            corner_radius=10,
            command=self.toggle_video_pause
        )
        self.pause_button.grid(row=0, column=1, padx=5, pady=5)  

        self.quit_button = ctk.CTkButton(
            self,
            text="Quit",
            width=90,
            height=45,
            fg_color='grey',
            corner_radius=10,
            command=self.close
        )
        self.quit_button.grid(row=8, column=0, padx=10, pady=10)  

        # self.slider_label = ctk.CTkLabel(
        #     self, 
        #     text="Jak mocny kolor?",  # Tekst etykiety
        #     font=("Arial", 14)  # Opcjonalnie zmień czcionkę i rozmiar
        # )
        # self.slider_label.grid(row=8, column=1, padx=10, pady=5)  # Umieszczenie etykiety

        # Alpha slider
        self.alpha_slider = ctk.CTkSlider(
            self,
            from_=0.1,
            to=0.7,
            number_of_steps=7,
            command=self.set_alpha,
            fg_color='pink',
            button_color='red'
        )
        self.alpha_slider.set(self.alpha)  
        self.alpha_slider.grid(row=7, column=1, padx=20, pady=10)

        color_frame = ctk.CTkFrame(self, fg_color="transparent")
        color_frame.grid(row=5, column=1, rowspan=4, padx=10, pady=10)

        # Buttons for changing color
        def change_color(color):
            self.color = color

        colors = [
            (200, 3, 200), (84, 7, 16), (219, 86, 139),
            (186, 11, 49), (33, 5, 5), (230, 138, 203),
            (99, 39, 52), (219, 68, 57), (207, 31, 69)
        ]
        for idx, color in enumerate(colors):
            ctk.CTkButton(
                color_frame,  # Użycie kontenera jako rodzica
                text=f"{idx}",
                fg_color=rgb_to_hex(color),
                width=60,
                height=60,
                corner_radius=30,
                command=lambda c=color: change_color([c[2], c[1], c[0]])
            ).grid(row=idx // 3, column=idx % 3, padx=5, pady=5) 

        # Start updating video
        self.update_video()

    def update_video(self):
        if not self.video_paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Crop the frame to the desired size
                height, width = frame.shape[:2]
                x_start = (width - self.desired_size) // 2
                y_start = (height - self.desired_size) // 2
                x_end = x_start + self.desired_size
                y_end = y_start + self.desired_size
                cropped_frame = frame[y_start:y_end, x_start:x_end]

                # Apply mask and overlay it on the frame
                mask = self.get_pred_from_frame(cropped_frame)
                masked_image = self.get_mask_on_photo(cropped_frame, mask, self.color, self.alpha)
                frame[y_start:y_end, x_start:x_end] = masked_image

                # Convert frame to RGB and update the video label
                self.current_frame = frame.copy()  # Store the current frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.configure(image=img)
                self.video_label.image = img

        # Schedule the next frame update
        self.after(10, self.update_video)

    def set_alpha(self, value):
        self.alpha = float(value)

    def toggle_video_pause(self):
        self.video_paused = not self.video_paused
        print("Video paused" if self.video_paused else "Video resumed")

    def save_photo(self):
        if self.current_frame is not None:
            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.png"
            folder = "photos"
            os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
            filepath = os.path.join(folder, filename)

            # Save the current frame
            cv2.imwrite(filepath, self.current_frame)
            print(f"Photo saved: {filepath}")

    def get_pred_from_frame(self, frame):
        image = self.preprocess_img_from_frame(frame)
        image = image.unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(image)

        output_image = output.squeeze(0).cpu().numpy()
        output_image = (output_image > 0.5).astype(np.uint8)
        output_image = np.transpose(output_image, (1, 2, 0))

        return output_image

    def preprocess_img_from_frame(self, frame):
        # Resize and normalize image
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.as_tensor(frame, dtype=torch.float) / 255.0
        frame = frame.permute(2, 0, 1)
        return frame

    def get_mask_on_photo(self, img, pred, color, alpha):
        mask = np.zeros_like(img)
        mask[:, :, 0] = pred[:, :, 0] * color[0]
        mask[:, :, 1] = pred[:, :, 0] * color[1]
        mask[:, :, 2] = pred[:, :, 0] * color[2]

        masked_image = img.copy()
        mask_indices = pred[:, :, 0] > 0

        if np.any(mask_indices):
            masked_image[mask_indices] = cv2.addWeighted(
                img[mask_indices].astype(np.float32),
                1 - alpha,
                mask[mask_indices].astype(np.float32),
                alpha,
                0,
            ).astype(np.uint8)

        return masked_image

    def close(self):
        self.cap.release()
        self.destroy()


# Main function to launch the app
if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    
    def start_main_app():
        welcome_screen.destroy()
        app = MaskApp(model_path=model_path)
        app.mainloop()
    
    model_path = r'C:\Users\aleks\OneDrive\Documents\LIPS SEGMENTATION\vgg_models\model_3_11_e5.pth'
    
    welcome_screen = WelcomeScreen(on_continue_callback=start_main_app)
    welcome_screen.mainloop()