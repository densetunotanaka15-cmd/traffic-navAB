
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os # Import os module to check for model existence
from ultralytics import YOLO
from gtts import gTTS # Import gTTS
from streamlit_tts import st_tts # Import streamlit_tts

st.set_page_config(page_title="Traffic Light Detector", layout="centered")

st.title("🚦 Traffic Light Detector for Visually Impaired")
st.markdown("Upload an image or use your camera to detect traffic light signals.")

# Sidebar for future options or for displaying information
st.sidebar.header("Application Settings")

# Accessibility settings
with st.sidebar.expander("Accessibility Settings"):
    tts_language = st.selectbox("Select TTS Language", options=['en', 'ja'], index=1) # Default to 'ja'
    enable_audio = st.checkbox("Enable Audio Feedback", value=True)
    st.write("Configure audio feedback and language for accessibility.")

# 2. Load the trained YOLOv8 model
# Make sure the path to the model weights is correct.
# The model 'best.pt' is usually saved in 'runs/detect/trainX/weights/' directory.
model_path = '/content/runs/detect/train2/weights/best.pt'

# Check if the model exists before loading
if not os.path.exists(model_path):
    st.error(f"Error: Trained model not found at {model_path}. Please ensure the model is trained and the path is correct.")
    st.stop() # Stop the app if model is not found

try:
    model = YOLO(model_path)
    st.sidebar.success("YOLOv8 model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading YOLOv8 model: {e}")
    st.stop()


# Image Upload Section
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image', use_column_width=True)
    st.success("Image uploaded successfully!")

    st.subheader("Detection Results")

    # 3. Perform inference
    try:
        # Convert PIL Image to OpenCV format (BGR numpy array)
        image_np = np.array(image_pil)
        # Convert RGB to BGR for OpenCV compatibility if needed (PIL is RGB)
        if image_np.shape[2] == 3: # Check if it's a color image
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np # Grayscale or other format

        # Perform inference
        results = model(image_cv) # model expects BGR

        # Process results and draw bounding boxes
        detected_lights_info = []
        processed_image = image_np.copy() # Start with the original RGB image for drawing

        for r in results:
            boxes = r.boxes # Bounding box information
            for box in boxes:
                # Get box coordinates in pixels
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get confidence score
                conf = round(float(box.conf[0]), 2)

                # Get class label
                cls = int(box.cls[0])
                label = model.names[cls]

                detected_lights_info.append(f"{label} (Confidence: {conf})")

                # Draw rectangle (BGR color, thickness)
                color = (0, 255, 0) # Green color for boxes (RGB)
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)

                # Put label and confidence
                text = f'{label} {conf}'
                cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the processed image
        st.image(processed_image, caption='Detected Traffic Lights', use_column_width=True)

        # 5. Update the 'Detected Traffic Lights' section and provide audio feedback
        if detected_lights_info:
            st.success("Traffic lights detected:")
            feedback_text = ""
            for light in detected_lights_info:
                st.write(f"- {light}")
                # Create audio feedback text
                if tts_language == 'ja':
                    if 'Red' in light: feedback_text += "赤信号です。 "
                    elif 'Green' in light: feedback_text += "青信号です。 "
                    elif 'Blue' in light: feedback_text += "青信号です。 " # Assuming blue means go or similar to green
                else:
                    feedback_text += f"Detected {light.split(' ')[0]} light. "

            if enable_audio and feedback_text:
                tts = gTTS(text=feedback_text, lang=tts_language)
                tts_audio_path = "tts_output.mp3"
                tts.save(tts_audio_path)
                st_tts(tts_audio_path, tts_language)

        else:
            st.info("No traffic lights detected in the image.")
            if enable_audio:
                no_detection_text = "信号機は検出されませんでした。" if tts_language == 'ja' else "No traffic lights detected."
                tts = gTTS(text=no_detection_text, lang=tts_language)
                tts_audio_path = "tts_output.mp3"
                tts.save(tts_audio_path)
                st_tts(tts_audio_path, tts_language)

    except Exception as e:
        st.error(f"An error occurred during inference: {e}")

else:
    st.info("Please upload an image to get started.")

# Camera Input Section (Placeholder)
st.header("Live Camera Feed (Future Feature)")
st.warning("Live camera input functionality will be implemented in a future update.")
st.write("You will be able to use your device's camera to detect traffic lights in real-time.")

st.markdown("--- ")
st.markdown("Developed for visually impaired individuals using YOLOv8.")
