import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import base64
from ultralytics import YOLO
from gtts import gTTS

# --- 設定 ---
st.set_page_config(page_title="信号機アシスタント", layout="centered")

# モデルの読み込みパスを修正（GitHub上のファイル名を直接指定）
# /content/... という記述を消して 'best.pt' だけにします
model_path = 'best.pt'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# --- UI (巨大な文字とボタン) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%; height: 100px;
        font-size: 30px !important; font-weight: bold;
        background-color: #0056b3; color: white; border-radius: 15px;
    }
    p, span, label { font-size: 24px !important; }
    .stAlert p { font-size: 32px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def announce(text, lang='ja'):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_b64 = base64.b64encode(fp.read()).decode()
        audio_tag = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
        st.markdown(audio_tag, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"音声エラー: {e}")

st.title("🚦 信号機アナウンサー")

if model is None:
    st.error(f"エラー: '{model_path}' が見つかりません。GitHubにファイルをアップロードしているか確認してください。")
    st.stop()

lang_code = st.selectbox("言語 / Language", ["ja", "en"])
is_jp = (lang_code == "ja")

uploaded_file = st.file_uploader("信号機の画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    results = model.predict(source=img, conf=0.4)
    
    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    reds = labels.count('Red')
    greens = labels.count('Green')
    
    if is_jp:
        if reds == 0 and greens == 0:
            msg = "信号機は見つかりませんでした。"
        else:
            msg = f"赤が{reds}個、青が{greens}個あります。"
            msg += " 青信号です。進めます。" if greens > 0 else " 赤信号です。止まってください。"
    else:
        if reds == 0 and greens == 0:
            msg = "No traffic lights detected."
        else:
            msg = f"Found {reds} red and {greens} green."
            msg += " It is green. You can go." if greens > 0 else " It is red. Please stop."

    if greens > 0:
        st.success(f"✅ {msg}")
    else:
        st.error(f"🛑 {msg}")
    
    announce(msg, lang_code)
    st.image(results[0].plot(), caption="検出結果", use_container_width=True)

