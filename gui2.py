import customtkinter as ctk
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from model_Unet import UNET
import os
from datetime import datetime
from tkinter import colorchooser
import random 
import tkinter as tk

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def bgr_to_hex(bgr):
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

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
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")


class EndScreen(ctk.CTk):
    def __init__(self, color='red'):
        super().__init__()
        self.title("Chosen Lipstick!")
        self.geometry("600x500")
        self.center_window(650, 500)
        self.color = bgr_to_hex(tuple(color))

        self.current_frame = None

        # canvas is needed for confetti
        self.canvas = tk.Canvas(self, bg="#1a1a1a", highlightthickness=0)  
        self.canvas.pack(fill="both", expand=True)

        
        hurra_label = ctk.CTkLabel(
            self,
            text="Your dream lipstick color!",
            font=("Lucida Handwriting", 20),
            text_color="white",
            justify="center"
        )
        hurra_label.pack(pady=20)

        # button - circle with choosed color
        main_button = ctk.CTkButton(
            self,
            text="",
            width=200,
            height=200,
            corner_radius=100,
            fg_color=self.color,
            text_color="white"
        )
        main_button.pack(pady=10)

        # buttons
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=10)

        quit_button = ctk.CTkButton(
            bottom_frame,
            text="Exit",
            width=100,
            height=50,
            corner_radius=10,
            fg_color="#7b1414",
            text_color="white",
            command=self.close
        )
        quit_button.pack(side="right", padx=10)

        back_button = ctk.CTkButton(
            bottom_frame,
            text="Back",
            width=100,
            height=50,
            corner_radius=10,
            fg_color="grey",
            text_color="white",
            command=self.back
        )
        back_button.pack(side="left", padx=10)

        save_button = ctk.CTkButton(
            bottom_frame,
            text="Save",
            width=100,
            height=50,
            corner_radius=10,
            fg_color='black', #"#ff39b3",
            text_color="white",
            command=lambda: self.save_color(color)
        )
        save_button.pack(side="top", padx=10)

        # generate confetti
        self.after(100, self.generate_confetti)

    def close(self):
        self.destroy()
    
    # function for going back to main app
    def back(self):
        self.destroy()
        app = MaskApp(model_path=model_path)
        app.mainloop()

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def generate_confetti(self):
        self.update_idletasks()  
        width = self.winfo_width()
        height = self.winfo_height()

        for _ in range(100):  # Liczba cząsteczek konfetti
            x = random.randint(0, width)  # Pozycja startowa X
            y = random.randint(-300, 0)  # Start powyżej okna
            size = random.randint(7, 15)  # Rozmiar konfetti
            color = random.choice(["#FFFFFF", "#FFC0CB", "#FF69B4", "#FF1493", "#FF0000"])
            self.animate_confetti(x, y, size, color)

    def animate_confetti(self, x, y, size, color):
        confetti = self.canvas.create_oval(x, y, x + size, y + size, fill=color, outline="")
        def move():
            nonlocal y
            y += random.randint(5, 10)  # Szybkość spadania
            x_offset = random.randint(-2, 2)  # Lekki ruch na boki
            self.canvas.move(confetti, x_offset, random.randint(5, 10))  # Poruszanie w dół i lekko w bok
            if y < self.winfo_height():  
                self.after(50, move)
            else:
                self.canvas.delete(confetti)  
        move()

    # function for saving choosed color 
    def save_color(self, color):
        folder = "colors"
        os.makedirs(folder, exist_ok=True)  
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"colors_{timestamp}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'a') as f:
            f.write(f"{color}\n")
        print(f"Color saved to {filepath}")


class MaskApp(ctk.CTk):
    def __init__(self, model_path, color=[39, 2, 255], device='cpu'):
        super().__init__()
        self.title("Onllipstick app")
        self.geometry("700x600")
        self.center_window(700, 600)
        self.desired_size = 256

        self.video_paused = False  
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model = self.model.to(device)
        self.model.eval()

        self.color = color
        self.alpha = 0.3 

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None  

        # Logo 
        self.logo_image = Image.open("logo2.png")  
        self.logo_image = self.logo_image.resize((400, 200))  
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        self.logo_label = ctk.CTkLabel(self, image=self.logo_photo, text="")
        self.logo_label.grid(row=0, column=1, columnspan=2, pady=(10,0))

        # Video
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=5, column=0, padx=40, pady=30)

        # Buttons
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
            text="Pause Video",
            fg_color='black',
            width=120,
            height=50,
            corner_radius=10,
            command=self.toggle_video_pause
        )
        self.pause_button.grid(row=0, column=1, padx=5, pady=5)  

        self.quit_button = ctk.CTkButton(
            self,
            text="Exit",
            width=90,
            height=45,
            fg_color='#7b1414',
            corner_radius=10,
            command=self.close
        )
        self.quit_button.grid(row=8, column=1, padx=10, pady=10) 

        button_frame2 = ctk.CTkFrame(self, fg_color="transparent")
        button_frame2.grid(row=8, column=0, pady=10)  

        self.custom_color_button = ctk.CTkButton(
            button_frame2,
            text="I don't like this colors!",
            fg_color='#9b004a',
            width=150,
            height=50,
            corner_radius=10,
            command=self.open_color_palette,
            font=('Open Sans',10)
        )
        self.custom_color_button.grid(row=0, column=0, padx=5, pady=5)

        self.custom_color_button = ctk.CTkButton(
            button_frame2,
            text="This is the perfect one!",
            fg_color='#e1026c',
            width=150,
            height=50,
            corner_radius=10,
            command=self.open_end_widnow, 
            font=('Open Sans',10)
        )
        self.custom_color_button.grid(row=0, column=2, padx=5, pady=5)

        # Colors Buttons
        color_frame = ctk.CTkFrame(self, fg_color="transparent")
        color_frame.grid(row=3, column=1, rowspan=4, padx=10, pady=10)

        def change_color(color):
            self.color = color

        colors = [
            (200, 3, 200), (84, 7, 16), (219, 86, 139),
            (186, 11, 49), (33, 5, 5), (230, 138, 203),
            (99, 39, 52), (219, 68, 57), (207, 31, 69)
        ]
        for idx, color in enumerate(colors):
            ctk.CTkButton(
                color_frame,  
                text=f"{idx}",
                fg_color=rgb_to_hex(color),
                width=60,
                height=60,
                corner_radius=30,
                command=lambda c=color: change_color([c[2], c[1], c[0]])
            ).grid(row=idx // 3, column=idx % 3, padx=5, pady=5) 

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
        self.alpha_slider.grid(row=7, column=1, padx=10, pady=5)

        self.update_video()

    def open_color_palette(self):
        color_code = colorchooser.askcolor(title="Wybierz kolor szminki")[0]
        if color_code:  
            r, g, b = map(int, color_code)
            self.color = [b, g, r]  
            print(f"Wybrany kolor: RGB({r}, {g}, {b})")

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

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
                self.current_frame = frame.copy() 
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.configure(image=img)
                self.video_label.image = img

        self.after(10, self.update_video)

    def set_alpha(self, value):
        self.alpha = float(value)

    def toggle_video_pause(self):
        self.video_paused = not self.video_paused
        print("Video paused" if self.video_paused else "Video resumed")

    def save_photo(self):
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.png"
            folder = "photos"
            os.makedirs(folder, exist_ok=True)  
            filepath = os.path.join(folder, filename)

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
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.as_tensor(frame, dtype=torch.float) / 255.0
        frame = frame.permute(2, 0, 1)
        return frame

    def get_mask_on_photo(self, img, pred, color, alpha):
        mask = np.zeros_like(img)
        mask[:, :, 0] = pred[:, :, 0] * color[0]  # Kanał R
        mask[:, :, 1] = pred[:, :, 0] * color[1]  # Kanał G
        mask[:, :, 2] = pred[:, :, 0] * color[2]  # Kanał B

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

    def open_end_widnow(self):
        self.destroy()
        end = EndScreen(color=self.color)
        end.mainloop()

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
    