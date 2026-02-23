import cv2
import numpy as np
from keras.models import load_model
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import webbrowser
import time
import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def resource_path(relative_path):
    """Get absolute path to resource for PyInstaller"""
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")  # When running normally
    return os.path.join(base_path, relative_path)


# ---------------- SETTINGS ---------------- #
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MOOD_TO_PLAYLIST = {
    'Happy': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
    'Sad': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
    'Angry': 'https://open.spotify.com/playlist/37i9dQZF1DWYp5sAHdz27Q',
    'Neutral': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO'
}

# -------------- INIT ---------------- #
cascade_path = resource_path("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

model_path = resource_path("emotion_model.h5")
model = load_model(model_path, compile=False)



# Spotify Auth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="b9d8b1061e5048cd97ed792b61e258ac",
    client_secret="e6dacfd699b94b7db6ee7fe257183b64T",
    redirect_uri="http://localhost:8888/callback",
    scope="user-read-playback-state,user-modify-playback-state"
))

# ----------- EMOTION DETECTION ----------- #
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        # Crop face
        roi_gray = gray[y:y+h, x:x+w]

        # Resize exactly to 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize & reshape
        roi = roi_gray.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)   # (48,48,1)
        roi = np.expand_dims(roi, axis=0)    # (1,48,48,1)

        # Predict
        prediction = model.predict(roi, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(prediction)]

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return emotion

    return None


# ----------- PLAY MUSIC ON SPOTIFY ----------- #
def play_music_for_mood(mood):
    if mood not in MOOD_TO_PLAYLIST:
        print(f"No playlist mapped for mood: {mood}")
        return
    url = MOOD_TO_PLAYLIST[mood]
    print(f"Detected mood: {mood}. Playing: {url}")
    webbrowser.open(url)

# -------------- MAIN FUNCTION -------------- #
def main():
    cap = cv2.VideoCapture(0)
    detected_mood = None
    mood_timer = time.time()

    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion(frame)

        if emotion:
            if time.time() - mood_timer > 5:  # Every 5 sec
                if emotion != detected_mood:
                    detected_mood = emotion
                    play_music_for_mood(emotion)
                mood_timer = time.time()

        cv2.imshow('Moodify - Mood Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ RUN ------------------ #
if __name__ == "__main__":
    main()
