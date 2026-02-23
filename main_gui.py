# main_gui.py — Full iPhone-style GUI Moodify App

import os, sys, time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import numpy as np
from keras.models import load_model
import webbrowser
import logging

# -------- PyInstaller resource fix --------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # PyInstaller temp dir
    except Exception:
        base_path = os.path.abspath(".")  # Normal run
    return os.path.join(base_path, relative_path)

# -------- Disable TensorFlow logs --------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -------- Emotion Settings --------
EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
MOOD_TO_PLAYLIST = {
    'Happy': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
    'Sad': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
    'Angry': 'https://open.spotify.com/playlist/37i9dQZF1DWYp5sAHdz27Q',
    'Neutral': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO'
}

# -------- Load Model & Cascade safely --------
model = load_model(resource_path("emotion_model.h5"), compile=False)
cascade = cv2.CascadeClassifier(resource_path("haarcascade_frontalface_default.xml"))

# -------- Rounded Webcam Frame Helper --------
def rounded_image(img_pil, size, radius=24):
    img = img_pil.resize(size, Image.LANCZOS)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0,0,size[0],size[1]), radius=radius, fill=255)
    img.putalpha(mask)
    return img

# -------- Main iPhone-Style GUI Class --------
class MoodifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Moodify — Emotion Based Music Player")

        try:
            self.root.iconbitmap(resource_path("assets/icon.ico"))
        except:
            pass

        self.root.geometry("980x620")
        self.root.configure(bg="#f7f8fc")

        # Main Wrapper
        main = tk.Frame(root, bg="#f7f8fc", padx=20, pady=20)
        main.pack(fill="both", expand=True)

        # Title
        tk.Label(main, text="Moodify",
                 font=("Helvetica", 26, "bold"),
                 bg="#f7f8fc").pack(anchor="w")

        content = tk.Frame(main, bg="#f7f8fc")
        content.pack(fill="both", expand=True, pady=(15,0))

        # Left Side — Webcam
        left = tk.Frame(content, bg="#f7f8fc")
        left.pack(side="left", fill="both", expand=True)

        # Webcam Preview Size
        self.preview_size = (580, 430)

        # Placeholder Image
        ph = Image.new("RGBA", self.preview_size, (240,240,240,255))
        ph_tk = ImageTk.PhotoImage(ph)

        self.preview = tk.Label(left, image=ph_tk, bg="#f7f8fc")
        self.preview.image = ph_tk
        self.preview.pack(pady=10)

        # Emotion Label Chip
        self.emotion_text = tk.Label(
            left,
            text="Detecting...",
            bg="#222",
            fg="white",
            font=("Helvetica", 14, "bold"),
            padx=14, pady=6
        )
        self.emotion_text.pack(pady=10)

        # Right Side — Playlist card + Button
        right = tk.Frame(content, bg="#f7f8fc")
        right.pack(side="right", fill="y", padx=20)

        # Card Art (replace with your own artwork)
        card_img = Image.open(resource_path("assets/card_art.png"))
        card_img = card_img.resize((260, 260))
        card_tk = ImageTk.PhotoImage(card_img)

        tk.Label(right, image=card_tk, bg="#f7f8fc").pack()
        self.card_img_ref = card_tk  # Prevent garbage collection

        # Play Button
        self.play_btn = tk.Button(
            right,
            text="Open Playlist",
            font=("Helvetica", 14, "bold"),
            bg="#222",
            fg="white",
            padx=20, pady=10,
            command=self.open_playlist
        )
        self.play_btn.pack(pady=20)

        # Start Webcam
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    # -------- Webcam Frame Updater --------
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)

            emotion = "No Face"
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48,48))
                face_norm = face_resized.astype("float32")/255.0
                face_norm = np.reshape(face_norm, (1,48,48,1))

                pred = model.predict(face_norm)[0]
                emotion = EMOTIONS[np.argmax(pred)]

                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # Update emotion label UI
            self.emotion_text.config(text=f"Emotion: {emotion}")

            # Convert → PIL → Rounded → Tkinter
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            rounded = rounded_image(pil, self.preview_size, radius=26)
            tk_img = ImageTk.PhotoImage(rounded)

            self.preview.config(image=tk_img)
            self.preview.image = tk_img

        self.root.after(10, self.update_frame)

    # -------- Open Playlist --------
    def open_playlist(self):
        emo = self.emotion_text.cget("text").replace("Emotion: ", "")
        if emo in MOOD_TO_PLAYLIST:
            webbrowser.open(MOOD_TO_PLAYLIST[emo])


# -------- Run App --------
if __name__ == "__main__":
    root = tk.Tk()
    app = MoodifyApp(root)
    root.mainloop()
