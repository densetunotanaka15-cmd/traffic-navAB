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
    /* ボタンを巨大化 */
    .stButton>button {
        width: 100%;
        height: 100px;
        font-size: 30px !important;
        font-weight: bold;
        background-color: #0056b3;
        color: white;
        border-radius: 15px;
    }
    /* テキストを大きく */
    p, span, label {
        font-size: 24px !important;
    }
    /* ステータス表示の巨大化 */
    .stAlert p {
        font-size: 32px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 音声再生関数 (base64でHTML埋め込み)
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

# モデルの読み込み
model_path = 'best.pt'
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# タイトル
st.title("🚦 信号機アナウンサー")

if model is None:
    st.error("モデルファイル(best.pt)が見つかりません。")
    st.stop()

# 言語設定（音声で切り替えを確認）
lang_code = st.selectbox("言語 / Language", ["ja", "en"])
is_jp = (lang_code == "ja")

# 起動時の挨拶
if 'initialized' not in st.session_state:
    start_msg = "信号機検出アプリが起動しました。カメラで撮影するか、画像をアップロードしてください。" if is_jp else "App started. Please take a photo or upload an image."
    announce(start_msg, lang_code)
    st.session_state.initialized = True

# 入力インターフェース
tab1, tab2 = st.tabs(["📸 カメラで撮影", "📂 画像を選択"])

with tab1:
    source_img = st.camera_input("信号機を撮影してください")

with tab2:
    uploaded_file = st.file_uploader("画像ファイルを選択", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        source_img = uploaded_file

# 推論とフィードバック
if source_img is not None:
    # 画像の読み込みと推論
    img = Image.open(source_img)
    results = model.predict(source=img, conf=0.4)
    
    # 検出結果の集計
    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    reds = labels.count('Red')
    greens = labels.count('Green')
    
    # メッセージ構築
    if is_jp:
        if reds == 0 and greens == 0:
            msg = "信号機は見つかりませんでした。もう一度試してください。"
        else:
            msg = f"赤信号が{reds}個、青信号が{greens}個あります。"
            if greens > 0:
                msg += " 青信号です。注意して進めます。"
            else:
                msg += " 赤信号です。止まってください。"
    else:
        if reds == 0 and greens == 0:
            msg = "No traffic lights detected. Please try again."
        else:
            msg = f"Found {reds} red and {greens} green lights."
            msg += " It is green. You can go." if greens > 0 else " It is red. Please wait."

    # 結果の表示（大きく表示）
    if greens > 0:
        st.success(f"✅ {msg}")
    else:
        st.error(f"🛑 {msg}")
    
    # 音声案内を実行
    announce(msg, lang_code)
    
    # 検出画像の表示
    res_img = results[0].plot()
    st.image(res_img, caption="検出結果の確認", use_container_width=True)

    # 再読み上げボタン（巨大）
    if st.button("もう一度音声を聞く"):
        announce(msg, lang_code)

st.markdown("---")
st.write("※このアプリは補助ツールです。必ず自身の耳と周囲の状況で安全を確認してください。")


