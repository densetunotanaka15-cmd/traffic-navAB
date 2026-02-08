import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import base64
from ultralytics import YOLO
from gtts import gTTS

# ページ設定: タイトルとアクセシビリティのためのスタイル適用
st.set_page_config(page_title="信号機アシスタント", layout="centered")

# CSSによるUIの巨大化とハイコントラスト設定
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 100px;
        font-size: 30px !important;
        font-weight: bold;
        background-color: #0056b3;
        color: white;
        border-radius: 15px;
    }
    p, span, label {
        font-size: 24px !important;
    }
    .stAlert p {
        font-size: 32px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 音声再生関数
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

# モデルの読み込みパスを修正
model_path = 'best.pt'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

st.title("🚦 信号機アナウンサー")

if model is None:
    st.error(f"モデルファイル({model_path})が見つかりません。GitHubに同名でアップロードされているか確認してください。")
    st.stop()

lang_code = st.selectbox("言語 / Language", ["ja", "en"])
is_jp = (lang_code == "ja")

if 'initialized' not in st.session_state:
    start_msg = "起動しました。画像をアップロードしてください。" if is_jp else "App started. Please upload an image."
    announce(start_msg, lang_code)
    st.session_state.initialized = True

uploaded_file = st.file_uploader("画像ファイルを選択", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # 推論実行
    results = model.predict(source=img, conf=0.4)
    
    # 検出結果の取得 (クラス名がRed, Green, Blueなどの想定)
    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    reds = sum(1 for label in labels if 'Red' in label)
    greens = sum(1 for label in labels if 'Green' in label or 'Blue' in label)
    
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
    
    res_img = results[0].plot()
    st.image(res_img, caption="検出結果", use_container_width=True)

    if st.button("もう一度音声を聞く"):
        announce(msg, lang_code)
