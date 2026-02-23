# app.py
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import time
import webbrowser
from PIL import Image
import base64
import io

# --------- SETTINGS ----------
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MOOD_TO_PLAYLIST = {
    'Happy': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
    'Sad': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
    'Angry': 'https://open.spotify.com/playlist/37i9dQZF1DWYp5sAHdz27Q',
    'Neutral': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO'
}
# -----------------------------

st.set_page_config(page_title="Moodify • Emotion Based Music", page_icon="🎧", layout="wide")
st.markdown("<style> .big-font {font-size:28px !important;} </style>", unsafe_allow_html=True)

# ---- Load model ----
@st.cache_resource(show_spinner=False)
def load_emotion_model(path="emotion_model.h5"):
    # model saved with older TF - use compile=False
    model = load_model(path, compile=False)
    return model

# Load model once
try:
    model = load_emotion_model("emotion_model.h5")
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    model_exception = e

# ---- Sidebar (navigation) ----
st.sidebar.image("assets/logo.png" if st.sidebar.button("Show logo") is False else "assets/logo.png", width=120) if st.sidebar.button("dummy") else None
st.sidebar.title("Moodify")
page = st.sidebar.radio("Navigation", ["Home", "Live Scan", "History", "Settings", "About"])

# Simple storage for session history
if "history" not in st.session_state:
    st.session_state.history = []

# ------- Home -------
if page == "Home":
    st.markdown("<h1 class='big-font'>Moodify — Emotion Based Music</h1>", unsafe_allow_html=True)
    st.write("A professional multi-device web app that detects your emotion through camera and suggests/plays mood-based Spotify playlists.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("### How it works")
        st.write("""
        1. Allow camera access.  
        2. App analyzes a frame and predicts emotion.  
        3. A playlist is suggested — open it to play.  
        """)
        st.write("### Quick start")
        st.info("Go to **Live Scan** and click **Start Scan**. Use the Settings page to change playlist links and model name.")
    with col2:
        if MODEL_OK:
            st.success("Model loaded successfully ✅")
        else:
            st.error("Model failed to load. Upload `emotion_model.h5` to the app folder.")
            st.exception(model_exception)

# ------- Live Scan -------
if page == "Live Scan":
    st.header("Live Mood Scan")
    if not MODEL_OK:
        st.error("Model is not available. Add emotion_model.h5 to the repo and reload.")
    else:
        st.write("Allow camera access and press **Capture Frame**. For best results, keep your face centered and well-lit.")
        camera_col, output_col = st.columns([1,1])

        with camera_col:
            camera_input = st.camera_input("Camera")
            capture_button = st.button("Capture Frame")
            auto_mode = st.checkbox("Auto-scan every 3 seconds", value=False)

        with output_col:
            result_placeholder = st.empty()
            bar = st.progress(0)

        def process_frame(img_bytes):
            # read image bytes to OpenCV
            file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ensure 48x48 input (model expects 48x48x1)
            roi = cv2.resize(gray, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))
            pred = model.predict(roi)[0]
            emotion_idx = int(np.argmax(pred))
            emotion = EMOTIONS[emotion_idx]
            confidence = float(pred[emotion_idx])
            return emotion, confidence, img

        last_mood = None
        last_time = 0

        if camera_input is not None:
            # show captured image
            img_bytes = camera_input.getvalue()
            st.image(img_bytes, caption="Captured frame", use_column_width=True)

            if capture_button:
                with st.spinner("Analyzing..."):
                    emo, conf, raw = process_frame(img_bytes)
                    st.session_state.history.insert(0, {"time": time.ctime(), "emotion": emo, "confidence": round(conf,3)})
                    result_placeholder.markdown(f"### Detected: **{emo}** — Confidence: `{conf:.2f}`")
                    bar.progress(int(conf * 100))
                    # draw bounding box & label on the image preview
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    display = np.array(pil)
                    # for demo: just show text label overlay
                    result_img = pil
                    st.image(result_img, caption=f"Result — {emo} ({conf:.2f})", use_column_width=True)

                    # Auto-open playlist if mapping exists and mood changed
                    if emo in MOOD_TO_PLAYLIST:
                        url = MOOD_TO_PLAYLIST[emo]
                        st.markdown(f"[Open playlist for **{emo}** ▶]({url})")
                        open_now = st.button("Open playlist in a new tab")
                        if open_now:
                            webbrowser.open_new_tab(MOOD_TO_PLAYLIST[emo])
                    else:
                        st.info("No playlist mapped for this emotion. Set one in Settings.")

        # Auto-scan loop (client must press camera input repeatedly on mobile; Auto mode is limited by browser)
        if auto_mode and camera_input is not None:
            st.info("Auto-scan enabled — press Capture Frame to start repeated scans (limited by browser constraints).")

# ------- History -------
if page == "History":
    st.header("Scan History")
    st.write("Shows last scans (stored in session only). Use Export to download a CSV.")
    if len(st.session_state.history) == 0:
        st.info("No scans yet — go to Live Scan and run one.")
    else:
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download history (CSV)", data=csv, file_name="mood_history.csv", mime="text/csv")

# ------- Settings -------
if page == "Settings":
    st.header("Settings")
    st.write("Customize playlist links, model file name and app preferences.")
    with st.form("settings_form"):
        for k in ['Happy', 'Sad', 'Angry', 'Neutral']:
            MOOD_TO_PLAYLIST[k] = st.text_input(f"Playlist for {k}", value=MOOD_TO_PLAYLIST.get(k, ""))
        model_name = st.text_input("Model filename", value="emotion_model.h5")
        save = st.form_submit_button("Save settings")
        if save:
            st.success("Settings saved locally for this session (persist in code / repo for permanent).")

    st.markdown("**Optional:** To enable Spotify playback control (open on user's Spotify device) you need to configure Spotify OAuth credentials and provide premium playback device. This is advanced and optional.")

    st.write("Advanced: If you want the app to remotely play on user's Spotify device, set up Client ID/Secret and integrate Spotipy on server side (requires OAuth).")

# ------- About -------
if page == "About":
    st.header("About Moodify")
    st.write("""
    Moodify is an emotion-aware music recommender that uses a Keras model trained on FER-2013 and a Streamlit web interface.
    This app is optimized for mobile and desktop browsers. For production, host the repo on Streamlit Cloud or any server that supports Streamlit.
    """)
    st.markdown("**Features**")
    st.markdown("- Camera input via browser\n- 48x48 emotion model\n- Session history and CSV export\n- Easy playlist mapping")
    st.markdown("---")
    st.markdown("Built with ❤️ — customize freely.")

# ------- Footer -------
st.sidebar.markdown("---")
st.sidebar.markdown("© Moodify — Built by you")
